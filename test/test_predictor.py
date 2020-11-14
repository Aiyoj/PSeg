if __name__ == "__main__":
    import cv2
    import numpy as np

    from pseg.predictor.u2net import U2NetPredictor

    model = U2NetPredictor({"resume": "models/new_u2netp.pth"})
    im = cv2.imread("test_images/pexels-photo-784754.png")
    pred = model.predict(im)

    cv2.imwrite("res24.jpg", (pred * 255).astype(np.uint8))
