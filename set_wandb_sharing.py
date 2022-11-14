import wandb

api_key = ''

def wandb_init(args):
    wandb.login(key=api_key)
    wandb.init(
        project='_dacon_breast_cancer',
        entity="mg_generation",
        name=args.work_dir.split('\\')[-1],
        tags=[args.img_model, args.tab_model, 'fold'+str(args.fold)],
        reinit=True,
        config=args.__dict__,
    )