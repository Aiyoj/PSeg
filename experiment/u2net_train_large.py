if __name__ == "__main__":
    import os

    from experiment.data_loader import DataLoader
    from pseg.algorithm.u2net import U2Net
    from pseg.modeling.loss.u2net_loss import U2NetLossV1, U2NetLossV2, U2NetLossV3
    from pseg.engine.optimizer_scheduler import OptimizerScheduler
    from pseg.engine.learning_rate import CosineDecayWithWarmupLearningRate, ConstantWithWarmupLearningRate
    from pseg.engine.model_saver import ModelSaver
    from pseg.engine.checkpoint import Checkpoint
    from pseg.utils.log import Logger
    from pseg.engine.trainer import Trainer

    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    epochs = 100
    lr = 0.001
    # save_interval = 331
    # log_interval = 331
    batch_size = 10

    model = U2Net({"model_name": "large"})

    criterion = U2NetLossV2()
    # criterion = U2NetLossV3()

    train_data_loader = DataLoader(
        data_dir=["/data2/xushiqi/seg_data/supervisely-person-datasets"],
        data_list=["/data2/xushiqi/seg_data/supervisely-person-datasets/train.txt"],
        batch_size=batch_size,
        num_worker=4,
        is_training=True,
        remainder=False,
        buffer_size=4,
    )

    save_interval = len(train_data_loader)
    log_interval = len(train_data_loader) // 2

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

    learning_rate = ConstantWithWarmupLearningRate(
        {
            "lr": lr,
            "epochs": epochs,
            "warmup_step": len(train_data_loader)
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
            "resume": "/data4/xushiqi/projects/PSeg/workspace/PSeg_BCE_Large_1119/u2net/model_epoch_38_minibatch_10830"
        }
    )

    logger = Logger(
        {
            "log_dir": "/data2/xushiqi/workspace",
            "verbose": False,
            "level": "info",
            "log_interval": log_interval,
            "name": "PSeg_BCE_Large_1119",
            "debug": False
        }
    )

    trainer = Trainer(
        model=model, criterion=criterion, optimizer_scheduler=optimizer_scheduler,
        learning_rate=learning_rate, model_saver=model_saver,
        checkpoint=checkpoint, logger=logger
    )
    trainer.train(data_loader=train_data_loader, epochs=epochs)
