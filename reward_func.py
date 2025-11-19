import numpy as np

def compare(im1:np.ndarray,im2:np.ndarray) -> float:
    assert im1.shape == im2.shape, "Images must have the same dimensions"
    assert im1.shape[2] == 2, "Images must have 2 channels, representing 2D vectors"
    num_pixels = im1.shape[0] * im1.shape[1]
    v1 = im1.reshape((num_pixels,2))
    v2 = im2.reshape((num_pixels,2))
    v1_mag = np.linalg.norm(v1, axis=1)
    v2_mag = np.linalg.norm(v2, axis=1)

    median = np.median(np.concatenate([v1_mag, v2_mag])) + 1e-6  # Avoid division by zero
    RI = np.zeros((v1.shape[0]))
    for i in range(v1.shape[0]):
        RI[i] = np.dot(v1[i],v2[i]) / (v1_mag[i] * v2_mag[i] + 1e-6)
    # RI = np.dot(v1,v2) / (v1_mag * v2_mag + 1e-6)
    # RI = np.einsum('ij,ij->i', v1, v2) / (v1_mag * v2_mag + 1e-6)
    WRI = ((1-RI)/2.0) * (v1_mag/np.median(v1_mag) + 1e-6) * (v2_mag/np.median(v2_mag) + 1e-6)
    WMI = np.abs(v1_mag - v2_mag) / median

    CMRI = (_normalize(WRI) + _normalize(WMI)) / 2.0
    return CMRI

def _normalize(v:np.ndarray) -> np.ndarray:
    len = v.shape[-1]
    sorted = np.sort(v, axis=-1)
    v_range = sorted[(int(len*0.02)):(int(len*0.98))]  # 2nd to 98th percentile
    high = np.max(v_range, axis=-1, keepdims=True)
    low = np.min(v_range, axis=-1, keepdims=True)

    return (v-low) / (high - low + 1e-6)

if __name__ == "__main__":
    im1 = np.zeros((100,100,2))
    im2 = np.zeros((100,100,2))
    im1[:,:,0] = 1.0
    im2[:,:,0] = 1.0
    im2[:,:,1] = 0.5
    im2[:,50:,1] = 1.0
    score = compare(im1,im2)
    print("CMRI score:", score)
    print("Mean CMRI score:", np.mean(score))