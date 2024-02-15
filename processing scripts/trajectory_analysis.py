import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def read_file_list(filename):
    """
    Reads a trajectory from a text file.

    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp.

    Input:
    filename -- File name

    Output:
    dict -- dictionary of (stamp,data) tuples

    """
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
            len(line) > 0 and line[0] != "#"]
    list = [(float(l[0]), l[1:]) for l in list if len(l) > 1]
    return dict(list)


def plot_traj(ax, stamps, traj, style, color, label):
    """
    Plot a trajectory using matplotlib.

    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend

    """
    stamps.sort()
    interval = np.median([s - t for s, t in zip(stamps[1:], stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i] - last < 2 * interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
        elif len(x) > 0:
            ax.plot(x, y, style, color=color, label=label)
            label = ""
            x = []
            y = []
        last = stamps[i]
    if len(x) > 0:
        ax.plot(x, y, style, color=color, label=label)


class TrajectoryAnalysis:
    def __init__(self, ground_truth_trajectory_path, comparison_trajectory_path,
                 data_labels=[],
                 output_directory_path=None,
                 plot_trajectory_name = None,
                 offset=0,
                 max_difference=0.0001,
                 scale=1.0,
                 verbose=True):

        if os.path.isfile(ground_truth_trajectory_path):
            self.ground_trajectory_path = ground_truth_trajectory_path
        else:
            raise ValueError("Invalid ground truth trajectory path")

        if os.path.isfile(comparison_trajectory_path):
            self.comparison_trajectory_path = comparison_trajectory_path
        else:
            raise ValueError("Invalid comparison trajectory path")

        if os.path.isdir(output_directory_path):
            self.output_path = output_directory_path
        else:
            raise ValueError("Invalid output path")

        self.ground_trajectory_dict = read_file_list(self.ground_trajectory_path)
        self.comparison_trajectory_dict = read_file_list(self.comparison_trajectory_path)

        # === Association parameters and variables ===
        self.offset = offset
        self.max_difference = max_difference
        self.matches = []
        self.associate()

        # Stamps
        self.first_stamps = list(self.ground_trajectory_dict.keys())  # Convert to list
        self.first_stamps.sort()

        self.second_stamps = list(self.comparison_trajectory_dict.keys())  # convert to list
        self.second_stamps.sort()

        # === Alignment parameters and variables ===
        self.verbose = verbose
        self.scale = scale
        self.rot = None
        self.trans = None
        self.trans_error = None
        self.ate_rmse = None

        self.first_xyz_full = None  # Full xyz trajectory
        self.second_xyz_full = None  # Full xyz trajectory

        self.first_xyz_matching = None  # Matching xyz trajectory
        self.second_xyz_matching = None  # Matching xyz trajectory

        self.second_xyz_aligned = None
        self.second_xyz_full_aligned = None
        self.perform_ate()

        # === output parameters and variables
        self.aligned_trajectory_path = self.output_path + "/analysis_aligned_trajectory.csv"
        self.aligned_associations_path = self.output_path + "/analysis_aligned_associations.csv"
        self.plot_trajectory_path = self.output_path + "/analysis_ate_trajectory.pdf"

        if isinstance(plot_trajectory_name, str):
            self.plot_trajectory_name = plot_trajectory_name
        else:
            self.plot_trajectory_name = None
        self.plot_error_path = self.output_path + "/analysis_ate_error.pdf"

        # Plotting Settings
        if isinstance(data_labels, list) and len(data_labels) == 2:
            self.data_labels = data_labels
        else:
            self.data_labels = ['Ground truth', 'Estimated trajectory']
        self.dr_color = 'r'
        self.gt_color = 'b'
        self.post_color = 'g'
        self.online_color = 'm'
        self.rope_color = 'b'
        self.buoy_color = 'k'
        self.title_size = 16
        self.legend_size = 12
        self.label_size = 14

    def associate(self, new_offset=None, new_max_difference=None):
        """
        Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
        to find the closest match for every input tuple.

        Input:
        first_list -- first dictionary of (stamp,data) tuples -> self.ground_trajectory_dict
        second_list -- second dictionary of (stamp,data) tuples -> self.comparison_trajectory_dict
        offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
        max_difference -- search radius for candidate generation

        Output:
        matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

        """

        if new_offset is not None:
            self.offset = new_offset

        if new_max_difference is not None:
            self.max_difference = new_max_difference

        # Ground truth trajectory
        first_keys = self.ground_trajectory_dict.keys()
        first_keys_list = list(first_keys)

        # Comparison trajectory
        second_keys = self.comparison_trajectory_dict.keys()
        second_keys_list = list(second_keys)

        potential_matches = [(abs(a - (b + self.offset)), a, b)
                             for a in first_keys_list
                             for b in second_keys_list
                             if abs(a - (b + self.offset)) < self.max_difference]

        potential_matches.sort()

        matches = []
        for diff, a, b in potential_matches:
            if a in first_keys_list and b in second_keys_list:
                # first_keys.remove(a)
                # second_keys.remove(b)
                # New method
                first_keys_list.remove(a)
                second_keys_list.remove(b)
                matches.append((a, b))

        matches.sort()
        self.matches = matches

        if len(self.matches) < 2:
            raise ValueError("No matches found")

    def align(self, model, data):
        """Align two trajectories using the method of Horn (closed-form).

        Input:
        model -- first trajectory (3xn)
        data -- second trajectory (3xn)

        Output:
        rot -- rotation matrix (3x3)
        trans -- translation vector (3x1)
        trans_error -- translational error per point (1xn)

        """
        np.set_printoptions(precision=3, suppress=True)
        model_zerocentered = model - model.mean(1)
        data_zerocentered = data - data.mean(1)

        W = np.zeros((3, 3))
        for column in range(model.shape[1]):
            W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
        U, d, Vh = np.linalg.linalg.svd(W.transpose())
        S = np.matrix(np.identity(3))
        if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
            S[2, 2] = -1
        rot = U * S * Vh
        trans = data.mean(1) - rot * model.mean(1)

        model_aligned = rot * model + trans
        alignment_error = model_aligned - data

        trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]

        return rot, trans, trans_error

    def perform_ate(self, scale=None, verbose=False):

        if scale is not None:
            self.scale = scale

        self.first_xyz_matching = np.matrix(
            [[float(value) for value in self.ground_trajectory_dict[a][0:3]] for a, b in self.matches]).transpose()

        self.second_xyz_matching= np.matrix(
            [[float(value) * float(self.scale) for value in self.comparison_trajectory_dict[b][0:3]] for a, b in
             self.matches]).transpose()

        # Determine alignment
        self.rot, self.trans, self.trans_error = self.align(self.second_xyz_matching, self.first_xyz_matching)

        self.first_xyz_full = np.matrix(
            [[float(value) for value in self.ground_trajectory_dict[b][0:3]] for b in self.first_stamps]).transpose()

        self.second_xyz_full = np.matrix(
            [[float(value) * float(self.scale) for value in self.comparison_trajectory_dict[b][0:3]] for b in
             self.second_stamps]).transpose()

        # perform alignment
        self.second_xyz_aligned = self.rot * self.second_xyz_matching + self.trans  # second_xyz_aligned
        self.second_xyz_full_aligned = self.rot * self.second_xyz_full + self.trans

        self.ate_rmse = np.sqrt(np.dot(self.trans_error, self.trans_error) / len(self.trans_error))

        if verbose:
            print(f"compared_pose_pairs {len(self.trans_error)} pairs")

            print(
                f"absolute_translational_error.rmse {self.ate_rmse} m")
            print(f"absolute_translational_error.mean {np.mean(self.trans_error)} m")
            print(f"absolute_translational_error.median {np.median(self.trans_error)} m")
            print(f"absolute_translational_error.std {np.std(self.trans_error)} m")
            print(f"absolute_translational_error.min {np.min(self.trans_error)} m")
            print(f"absolute_translational_error.max {np.max(self.trans_error)} m")
        else:
            print(self.ate_rmse)

    def plot_trajectories(self, title=None, plot_name=None):

        if plot_name is not None and isinstance(plot_name, str):
            self.plot_trajectory_name = plot_name

        # matplotlib.use('Agg') call before import matplotlib.pyplot ...
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_traj(ax, self.second_stamps, self.second_xyz_full.transpose().A,
                  '-', "red", label=self.data_labels[1])

        plot_traj(ax, self.first_stamps, self.first_xyz_full.transpose().A,
                  '-', "green", label=self.data_labels[0])

        plot_traj(ax, self.second_stamps, self.second_xyz_full_aligned.transpose().A,
                  '-', "blue", label=f"Aligned " + self.data_labels[1].casefold())

        label = "Difference"
        for (a, b), (x1, y1, z1), (x2, y2, z2) in zip(self.matches, self.first_xyz_matching.transpose().A,
                                                      self.second_xyz_aligned.transpose().A):
            ax.plot([x1, x2], [y1, y2], '-', color="orange", label=label, alpha=0.5)
            label = ""

        ax.legend()

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        if isinstance(title, str):
            ax.set_title(title + f"\nATE RMSE: {self.ate_rmse:.3f}")
        else:
            ax.set_title(f"ATE RMSE: {self.ate_rmse}")

        if isinstance(self.plot_trajectory_name, str):
            plt.savefig(self.output_path + f"/{self.plot_trajectory_name}.pdf", format="pdf")
        else:
            plt.savefig(self.plot_trajectory_path, format="pdf")

    def plot_errors(self, errors, error_titles, title=None, plot_name=None):
        # matplotlib.use('Agg') call before import matplotlib.pyplot ...

        if not isinstance(errors, list) or not isinstance(error_titles, list):
            raise TypeError("Please provide a list of translational errors and matching names")

        if len(errors) != len(error_titles) or len(errors) == 0:
            raise ValueError("Please provide at least one")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        rms_errors = []
        for error, error_info in zip(errors, error_titles):

            error_title = error_info[0]
            error_color = error_info[1]

            trans_rmse = np.sqrt(np.dot(error, error) / len(error))
            rms_errors.append(trans_rmse)

            ax.plot(error, '-', color=error_color, label=error_title + f", RMSE: {trans_rmse:3f} m")

        ax.legend()

        ax.set_xlabel('m_t')
        ax.set_ylabel('ATE [m]')

        if isinstance(title, str):
            ax.set_title(title + f"\nATE RMSE: {self.ate_rmse:.3f}")
        else:
            ax.set_title(f"ATE RMSE: {self.ate_rmse}")

        if isinstance(plot_name, str):
            plt.savefig(self.output_path + f"/{plot_name}.pdf", format="pdf")
        else:
            plt.savefig(self.plot_error_path, format="pdf")

