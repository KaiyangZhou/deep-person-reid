if __name__ == '__main__':
    import torchreid

    datamanager = torchreid.data.ImageDataManager(
        root='reid-data',
        sources='market1501',
        targets='market1501',
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=['random_flip', 'color_jitter']
    )

    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=datamanager.num_train_pids,
        loss='softmax',
        pretrained=True
    )
    model = model.cuda()

    weight_path = 'log\osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'
    torchreid.utils.load_pretrained_weights(model, weight_path)

    optimizer = torchreid.optim.build_optimizer(model, optim='adam', lr=0.0003)

    
    # scheduler = torchreid.optim.build_lr_scheduler(
    #  optimizer, lr_scheduler='single_step', stepsize=20
    # )
    

    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        # scheduler=scheduler,
        label_smooth=True
    )

    engine.run(
        save_dir='log/osnet_ibn_x1_0',
        # max_epoch=60,
        # eval_freq=10,
        # print_freq=10,
        test_only=True,
        visrank=True
    )
