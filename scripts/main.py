import torchreid

datamanager = torchreid.data.ImageDataManager(
    root='reid-data',
    sources='market1501',
    height=128,
    width=64,
    combineall=False,
    batch_size=16
)
model = torchreid.models.build_model(
    name='squeezenet1_0',
    num_classes=datamanager.num_train_pids,
    loss='softmax'
)
optimizer = torchreid.optim.build_optimizer(model)
scheduler = torchreid.optim.build_lr_scheduler(optimizer, lr_scheduler='single_step', stepsize=20)
engine = torchreid.engine.ImageSoftmaxEngine(datamanager, model, optimizer, scheduler=scheduler)
#engine.run(max_epoch=1, print_freq=1, fixbase_epoch=0, open_layers='classifier', test_only=True)