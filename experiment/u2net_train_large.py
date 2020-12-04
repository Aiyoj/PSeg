if __name__ == "__main__":
    import os

    from experiment.data_loader import DataLoader
    from pseg.algorithm.u2net import U2Net
    from pseg.modeling.loss.u2net_loss import U2NetLossV1, U2NetLossV2, U2NetLossV3, U2NetLossV4, U2NetLossV5, \
        U2NetLossV6
    from pseg.engine.optimizer_scheduler import OptimizerScheduler
    from pseg.engine.learning_rate import CosineDecayWithWarmupLearningRate, ConstantWithWarmupLearningRate, \
        ConstantLearningRate
    from pseg.engine.model_saver import ModelSaver
    from pseg.engine.checkpoint import Checkpoint
    from pseg.utils.log import Logger
    from pseg.engine.trainer import Trainer

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,6"

    epochs = 200
    lr = 0.001
    batch_size = 24

    model_config = {
        "backbone": {
            "type": "U2NetBackboneV3", "args": {"in_ch": 3, "model_name": "large"}
        },
        "head": {
            "type": "U2NetHeadV3", "args": {"out_ch": 1, "model_name": "large"}
        }
    }
    model = U2Net(model_config)

    criterion = U2NetLossV6()

    pre_processes = [
        {"type": "AugHSV", "args": {"p": 0.5}},
        # {"type": "AugNoise", "args": {"p": 0.5}},
        {"type": "AugGray", "args": {"p": 0.5}},
        {"type": "AugBlur", "args": {"p": 0.5}},
        # {"type": "AugMotionBlur", "args": {"p": 0.5}},
        {"type": "FlipLR", "args": {"p": 0.5}},
        {"type": "Affine", "args": {"rotate": [-30, 30]}},
        {"type": "RandomCropV3", "args": {"size1": (640, 640), "size2": (576, 576)}},
        {"type": "Perspective", "args": {"p": 0.5}},
        {"type": "ThinPlateSpline", "args": {"p": 0.5}},
        {"type": "Normalize", "args": {}},
        # {"type": "MakeDistMap", "args": {}}
    ]

    train_data_loader = DataLoader(
        data_dir=["/input0/LV-MHP-v2", "/input0/humanparsing-atr", "/input0/PortraitMatting"],
        data_list=["/input0/LV-MHP-v2/train.txt", "/input0/humanparsing-atr/train.txt",
                   "/input0/PortraitMatting/train.txt"],
        batch_size=batch_size,
        num_worker=4,
        is_training=True,
        remainder=False,
        buffer_size=4,
        pre_processes=pre_processes
    )

    save_interval = len(train_data_loader) * 2
    log_interval = len(train_data_loader)

    optimizer_scheduler = OptimizerScheduler(
        {
            "optimizer": "Adam",
            "lr": lr,
            "optimizer_args": {
                "lr": lr,
                # "momentum": 0.9,
                # "weight_decay": 0.00001
            }
        }
    )

    # learning_rate = CosineDecayWithWarmupLearningRate(
    #     {
    #         "lr": lr,
    #         "epochs": epochs,
    #         "step_each_epoch": len(train_data_loader),
    #         "warmup_step": len(train_data_loader)
    #     }
    # )

    # learning_rate = ConstantWithWarmupLearningRate(
    #     {
    #         "lr": lr,
    #         "epochs": epochs,
    #         "warmup_step": len(train_data_loader)
    #     }
    # )

    learning_rate = ConstantLearningRate(
        {
            "lr": lr
        }
    )

    model_saver = ModelSaver(
        {

            "dir_path": "u2net",
            "save_interval": save_interval,
            "signal_path": "save"
        }
    )

    checkpoint = Checkpoint(
        {
            "start_epoch": 0,
            "start_iter": 0,
            # "resume": "models/new_u2net.pth"
        }
    )

    logger = Logger(
        {
            "log_dir": "/output/workspace",
            "verbose": False,
            "level": "info",
            "log_interval": log_interval,
            "name": "GL_PSegV3_BCE_Large",
            "debug": False
        }
    )

    trainer = Trainer(
        model=model, criterion=criterion, optimizer_scheduler=optimizer_scheduler,
        learning_rate=learning_rate, model_saver=model_saver,
        checkpoint=checkpoint, logger=logger, frozen_bn=False
    )
    trainer.train(data_loader=train_data_loader, epochs=epochs)
