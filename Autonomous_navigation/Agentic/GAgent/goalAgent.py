class GoalAgent:
    def __init__(self, cost_ring_radius: DynamicCostRing):
        self.cost_ring = cost_ring

    def compute_relative_goal(self, current_pose, global_goal):
        """
        Computes the target goal relative to the robot's local coordinate frame.

        If global goal is beyond the cost ring radius, projects it onto the edge
        of the cost ring along the goal direction vector.

        :param current_pose: (x, y, yaw) in global frame.
        :param global_goal: (x, y) in global frame.
        :return: (rel_x, rel_y) local coordinates, distance_to_global_goal, is_intermediate
        """
        rx, ry, ryaw = current_pose
        gx, gy = global_goal

        # Global displacement vector
        dx = gx - rx
        dy = gy - ry
        dist_to_global = np.hypot(dx, dy)

        # Handle edge case where robot is already at the global goal
        if dist_to_global < 1e-5:
            return 0.0, 0.0, 0.0, False

        # Determine target point in global frame
        effective_radius = self.cost_ring.radius
        if dist_to_global > effective_radius:
            # Scale displacement vector down to cost ring boundary
            target_gx = rx + (dx / dist_to_global) * effective_radius
            target_gy = ry + (dy / dist_to_global) * effective_radius
            is_intermediate = True
        else:
            target_gx = gx
            target_gy = gy
            is_intermediate = False

        # Transform global target point to local robot frame (X-forward, Y-left)
        local_dx = target_gx - rx
        local_dy = target_gy - ry

        rel_x = local_dx * np.cos(ryaw) + local_dy * np.sin(ryaw)
        rel_y = -local_dx * np.sin(ryaw) + local_dy * np.cos(ryaw)

        return rel_x, rel_y, dist_to_global, is_intermediate
