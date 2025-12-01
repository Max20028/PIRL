import torch

def compare(im1:torch.Tensor,im2:torch.Tensor) -> torch.Tensor:
    assert im1.shape == im2.shape, "Images must have the same dimensions"
    assert im1.shape[-1] == 2, "Images must have 2 channels, representing 2D vectors"

    # TODO - make it support a batch dimension
    v1 = im1.view(-1, 2)
    v2 = im2.view(-1, 2)

    v1_mag = torch.linalg.norm(v1, dim=1)
    v2_mag = torch.linalg.norm(v2, dim=1)

    median = torch.median(torch.cat([v1_mag, v2_mag])) + 1e-6  # Avoid division by zero
    RI = torch.sum(v1 * v2, dim=1) / (v1_mag * v2_mag + 1e-6)
    WRI = ((1-RI)/2.0) * (v1_mag/torch.median(v1_mag) + 1e-6) * (v2_mag/torch.median(v2_mag) + 1e-6)
    WMI = torch.abs(v1_mag - v2_mag) / median

    CMRI = (_normalize(WRI) + _normalize(WMI)) / 2.0
    return CMRI

def _normalize(v:torch.Tensor) -> torch.Tensor:
    # len = v.shape[-1]
    # sorted = np.sort(v, axis=-1)
    # v_range = sorted[(int(len*0.02)):(int(len*0.98))]  # 2nd to 98th percentile
    # high = np.max(v_range, axis=-1, keepdims=True)
    # low = np.min(v_range, axis=-1, keepdims=True)

    high = torch.quantile(v, 0.98)
    low = torch.quantile(v, 0.02)

    return (v-low) / (high - low + 1e-6)

if __name__ == "__main__":
    im1 = torch.zeros((100,100,2))
    im2 = torch.zeros((100,100,2))
    im1[:,:,0] = 1.0
    im2[:,:,0] = 1.0
    im2[:,:,1] = 0.5
    im2[:,50:,1] = 1.0
    score = compare(im1,im2)
    print("CMRI score:", score)
    print("Mean CMRI score:", torch.mean(score))