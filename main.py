import torchreid

dataset = torchreid.data.ImageDatasetManager(
    root='data',
    source_names='grid',
    height=128,
    width=64
)
model = torchreid.models.build_model(
    name='squeezenet1_0',
    num_classes=dataset.num_train_pids,
    loss='softmax'
)
optimizer = torchreid.optim.build_optimizer(model)
scheduler = torchreid.optim.build_lr_scheduler(optimizer, lr_scheduler='multi_step', stepsize=[10, 20])
engine = torchreid.engine.ImageSoftmaxEngine(dataset, model, optimizer, scheduler=scheduler)
engine.run(max_epoch=3, print_freq=1, fixbase_epoch=3, open_layers='classifier')