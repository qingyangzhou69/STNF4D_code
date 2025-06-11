import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def plot_flow(flow, X=None, Y=None, stride=10):
    flow = flow.copy()
    # flow[:, :, 0] = -flow[:, :, 0]
    if X is None:
        height, width, _ = flow.shape
        xx = np.arange(0, height, stride)
        yy = np.arange(0, width, stride)
        X, Y = np.meshgrid(xx, yy)
        X = X.flatten()
        Y = Y.flatten()

        # sample
        sample_0 = flow[:, :, 0][xx]
        sample_0 = sample_0.T
        sample_x = sample_0[yy]
        sample_x = sample_x.T
        sample_1 = flow[:, :, 1][xx]
        sample_1 = sample_1.T
        sample_y = sample_1[yy]
        sample_y = sample_y.T

        sample_x = sample_x[:, :, np.newaxis]
        sample_y = sample_y[:, :, np.newaxis]
        new_flow = np.concatenate([sample_x, sample_y], axis=2)
    flow_x = new_flow[:, :, 0].flatten()
    flow_y = new_flow[:, :, 1].flatten()

    # display
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    # plt.quiver(X,Y, flow_x, flow_y, angles="xy", color="#666666")
    ax.quiver(X, Y, flow_x, flow_y, color="#666666")
    ax.grid()
    # ax.legend()
    plt.draw()
    plt.show()
def plot_rays(ray_directions: np.array, ray_origins: np.array, ray_length: float):
    """
    Plot rays of a scanner (open3d).

    Args:
    ray_directions (np.array(W, H, 3)): ray directions.
    ray_origins (np.array(W, H, 3)): ray origins.
    ray_length (float): ray length.

    Returns:
    lines (o3d.geometry.LineSet): output lines.

    """

    W, H, _ = ray_directions.shape
    ori1 = ray_origins[0, 0, :]
    ori2 = ray_origins[W - 1, 0, :]
    ori3 = ray_origins[W - 1, H - 1, :]
    ori4 = ray_origins[0, H - 1, :]
    end1 = ray_origins[0, 0, :] + ray_directions[0, 0, :] * ray_length
    end2 = ray_origins[W - 1, 0, :] + ray_directions[W - 1, 0, :] * ray_length
    end3 = ray_origins[W - 1, H - 1, :] + ray_directions[W - 1, H - 1, :] * ray_length
    end4 = ray_origins[0, H - 1, :] + ray_directions[0, H - 1, :] * ray_length
    lines = [[0, 4], [1, 5], [2, 6], [3, 7], [4, 5], [5, 6], [6, 7], [7, 4]]
    pts = np.vstack([ori1, ori2, ori3, ori4, end1, end2, end3, end4])
    line_ray = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )

    return line_ray

def plot_camera_pose(pose):
    """
    Plot camera pose (open3d).

    Args:
    pose (np.array(4, 4)): camera pose.

    Returns:
    lines (o3d.geometry.LineSet): output lines.

    """

    colorlines = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    origin = np.array([[0], [0], [0], [1]])
    axes = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1],
                     [1, 1, 1]])
    points = np.vstack([np.transpose(origin), np.transpose(axes)])[:, :-1]
    lines = [[0, 1], [0, 2], [0, 3]]
    worldframe = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    worldframe.colors = o3d.utility.Vector3dVector(colorlines)

    # bbox
    xyz_min = [-0.5, -0.5, -0.5]
    xyz_max = [0.5, 0.5, 0.5]
    points = [[xyz_min[0], xyz_min[1], xyz_min[2]],
              [xyz_max[0], xyz_min[1], xyz_min[2]],
              [xyz_min[0], xyz_max[1], xyz_min[2]],
              [xyz_max[0], xyz_max[1], xyz_min[2]],
              [xyz_min[0], xyz_min[1], xyz_max[2]],
              [xyz_max[0], xyz_min[1], xyz_max[2]],
              [xyz_min[0], xyz_max[1], xyz_max[2]],
              [xyz_max[0], xyz_max[1], xyz_max[2]]]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set_bbox = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set_bbox.colors = o3d.utility.Vector3dVector(colors)

    origin = np.array([[0], [0], [0], [1]])
    unit = 0.3
    axes = np.array([[unit, 0, 0],
                     [0, unit, 0],
                     [0, 0, unit],
                     [1, 1, 1]])
    axes_trans = np.dot(pose, axes)
    origin_trans = np.dot(pose, origin)
    points = np.vstack([np.transpose(origin_trans), np.transpose(axes_trans)])[:, :-1]
    lines = [[0, 1], [0, 2], [0, 3]]
    colorlines = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colorlines)

    return line_set + worldframe

def plot_cube(cube_center: np.array, cube_size: np.array):
    """
    Plot a cube (open3d).

    Args:
    cube_center (np.array(3, 1)): cube center.
    cube_size (np.array(3, 1)): cube size.

    Returns:
    lines (o3d.geometry.LineSet): output lines.

    """

    # coordinate frame
    colorlines = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    origin = np.array([[0], [0], [0], [1]])
    unit = 0.3
    axes = np.array([[unit, 0, 0],
                     [0, unit, 0],
                     [0, 0, unit],
                     [1, 1, 1]]) * np.vstack([np.hstack([cube_size, cube_size, cube_size]), np.ones((1, 3))])
    points = np.vstack([np.transpose(origin), np.transpose(axes)])[:, :-1]
    points += cube_center.squeeze()
    lines = [[0, 1], [0, 2], [0, 3]]
    worldframe = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    worldframe.colors = o3d.utility.Vector3dVector(colorlines)
    # bbox
    xyz_min = cube_center.squeeze() + np.array([-0.5, -0.5, -0.5]) * cube_size.squeeze()
    xyz_max = cube_center.squeeze() + np.array([0.5, 0.5, 0.5]) * cube_size.squeeze()
    points = [[xyz_min[0], xyz_min[1], xyz_min[2]],
              [xyz_max[0], xyz_min[1], xyz_min[2]],
              [xyz_min[0], xyz_max[1], xyz_min[2]],
              [xyz_max[0], xyz_max[1], xyz_min[2]],
              [xyz_min[0], xyz_min[1], xyz_max[2]],
              [xyz_max[0], xyz_min[1], xyz_max[2]],
              [xyz_min[0], xyz_max[1], xyz_max[2]],
              [xyz_max[0], xyz_max[1], xyz_max[2]]]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set_bbox = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set_bbox.colors = o3d.utility.Vector3dVector(colors)
    return line_set_bbox + worldframe