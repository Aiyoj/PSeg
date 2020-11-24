if __name__ == "__main__":
    import cv2
    import numpy as np

    from experiment.data_loader import DataLoader

    pre_processes = [
        {"type": "AugHSV", "args": {"p": 0.5}},
        {"type": "AugNoise", "args": {"p": 0.5}},
        {"type": "AugGray", "args": {"p": 0.5}},
        {"type": "AugBlur", "args": {"p": 0.5}},
        # {"type": "AugMotionBlur", "args": {"p": 0.5}},
        {"type": "FlipLR", "args": {"p": 0.5}},
        {"type": "Affine", "args": {"rotate": [-20, 20]}},
        {"type": "RandomCrop", "args": {}}
    ]

    dl = DataLoader(
        data_dir=["/data2/xushiqi/seg_data/supervisely-person-datasets"],
        data_list=["/data2/xushiqi/seg_data/supervisely-person-datasets/train.txt"],
        batch_size=1,
        num_worker=5,
        pre_processes=pre_processes
    )
    for i, k in enumerate(dl):
        print(0, i, k["image"].shape)
        print(0, i, k["label"].shape)
        print(0, i, k["filename"])
        image = cv2.cvtColor(k["image"][0].cpu().numpy(), cv2.COLOR_RGB2BGR)
        label = k["label"][0].cpu().numpy()

        cv2.imwrite("1.png", image)
        cv2.imwrite("3.png", label)

        m = label.copy().astype(np.uint8)

        red_mask = np.zeros(image.shape, dtype=np.uint8)
        red_mask[:, :, :] = np.array([0, 0, 255])

        idx1 = m >= 127
        idx2 = m < 127
        m[idx1] = 1
        m[idx2] = 0

        m = np.expand_dims(m, 2)

        dst = (image * m) * 0.7 + (red_mask * m) * 0.3

        dst = (image * (1 - m)) + dst

        cv2.imwrite("2.png", dst)

        break
