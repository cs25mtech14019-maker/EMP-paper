import torch
import numpy as np

def profile_batch_kinematics(y_future: torch.Tensor, dt: float = 0.1):
    """
    Computes the maximum acceleration and yaw rate for each sample in a batch.
    
    Args:
        y_future: Tensor of shape [B, 60, 2] representing 6s of future trajectories
        dt: The timedelta between steps (0.1s for 10Hz AV2 data)
        
    Returns:
        max_accel: Tensor of shape [B] containing the max acceleration (m/s^2) for each sample
        max_yaw_rate: Tensor of shape [B] containing the max yaw rate (rad/s) for each sample
    """
    # 1. Velocity (m/s)
    v = torch.diff(y_future, dim=1) / dt  # [B, 59, 2]
    
    # 2. Acceleration (m/s^2)
    a = torch.diff(v, dim=1) / dt         # [B, 58, 2]
    a_mag = torch.norm(a, dim=-1)         # [B, 58]
    max_accel_per_sample = a_mag.max(dim=-1)[0] # [B]
    
    # 3. Yaw Rate (rad/s)
    yaw = torch.atan2(v[..., 1], v[..., 0]) # [B, 59]
    yaw_diff = torch.diff(yaw, dim=1)       # [B, 58]
    
    # Strict handling for angle wrap-around (-pi to pi)
    yaw_diff = (yaw_diff + torch.pi) % (2 * torch.pi) - torch.pi
    yaw_rate = torch.abs(yaw_diff) / dt
    max_yaw_rate_per_sample = yaw_rate.max(dim=-1)[0] # [B]
    
    return max_accel_per_sample.cpu().numpy(), max_yaw_rate_per_sample.cpu().numpy()

if __name__ == "__main__":
    print("Profiling script loaded. To find exact 98th-percentile bounds for your training subset:")
    print("1. Inject profile_batch_kinematics into your LightningModule's training_step temporarily.")
    print("2. Accumulate the returned arrays over 100-500 batches.")
    print("3. Print: np.percentile(all_accels, 98) and np.percentile(all_yaws, 98)")
    print("For quick testing or if dataloader is heavy, we default strictly to physical limits (1.0g Accel).")
