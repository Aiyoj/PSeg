if __name__ == "__main__":
    import os
    import cv2
    import numpy as np

    from pseg.predictor.u2net import U2NetPredictor

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    model = U2NetPredictor(
        {
            "resume": "/data2/xushiqi/workspace/PSeg_BCE_Large_1119/u2net/model_epoch_100_minibatch_57100",
            "model_name": "large"
        }
    )

    # model = U2NetPredictor(
    #     {"resume": "models/new_u2net.pth", "model_name": "large"}
    # )
    im = cv2.imread("test_images/static_pics/images-718.jpg")
    # im = cv2.imread("test_images/pexels-photo-784754.png")
    # im = cv2.imread("test_images/bodybuilder-weight-training-stress-38630.png")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    pred = model.predict(im)

    cv2.imwrite("res25.jpg", (pred * 255).astype(np.uint8))
