if __name__ == "__main__":
    import os
    import cv2
    import zlib
    import json
    import base64
    import numpy as np

    all = os.walk("supervisely-person-datasets")

    image_paths = []
    for path, dir, filelist in all:
        if "mask" in path:
            continue
        for filename in filelist:
            if filename.endswith("jpg") or filename.endswith("png") or \
                    filename.endswith("JPG") or filename.endswith("jpeg") or \
                    filename.endswith("PNG") or filename.endswith("JPEG"):
                image_paths.append("{}/{}".format(path, filename))

    for image_path in image_paths:
        basename = os.path.basename(image_path)
        dir_path = os.path.dirname(image_path)
        save_dir = dir_path.replace("/img", "/mask")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        gt_path = image_path.replace("/img/", "/ann/") + ".json"

        im = cv2.imread(image_path)
        # print(image_path)
        h, w = im.shape[:2]

        mask = np.zeros(im.shape[:2], np.uint8)

        with open(gt_path, "rb") as f:
            json_data = json.load(f)

            exteriors = []
            interiors = []

            for object_data in json_data["objects"]:
                if object_data["bitmap"] is not None:
                    data = object_data["bitmap"]["data"]
                    origin = object_data["bitmap"]["origin"]
                    z = zlib.decompress(base64.b64decode(data))
                    image_array = np.frombuffer(z, np.uint8)
                    image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
                    image = image[:, :, 0]

                    idx = np.where(image > 127)
                    # mask[idx] = 255
                    mask[(idx[0] + origin[1], idx[1] + origin[0])] = 255
                else:
                    exterior = object_data["points"]["exterior"]
                    interior = object_data["points"]["interior"]
                    if len(exterior) > 0:
                        exteriors.append(np.array(exterior, dtype=np.int32))
                    if len(interior) > 0:
                        for i in interior:
                            interiors.append(np.array(i, dtype=np.int32))

            if len(exteriors) > 0:
                cv2.fillPoly(mask, pts=exteriors, color=255)
            if len(interiors) > 0:
                cv2.fillPoly(mask, pts=interiors, color=0)

            # print("{}/{}.png".format(save_dir, basename.split(".")[0]))
            cv2.imwrite("{}/{}.png".format(save_dir, basename.split(".")[0]), mask)
