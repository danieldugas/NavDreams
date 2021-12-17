conf = Namespace(action_dim=3, actor_dist='onehot', actor_grad='reinforce', adam_eps=1e-05, adam_lr=0.0003, adam_lr_actor=0.0001, adam_lr_critic=0.0001, amp=True, batch_length=50, batch_size=50, buffer_size=10000000, clip_rewards=None, cnn_depth=48, data_workers=1, deter_dim=2048, device='cuda:0', enable_profiler=False, entropy=0.003, env_action_repeat=1, env_id='NavRep3DStaticASLEnv', env_id_eval=None, env_no_terminal=False, env_time_limit=27000, eval_batch_size=10, eval_batches=61, eval_interval=2000, eval_samples=10, gamma=0.995, generator_prefill_policy='random', generator_prefill_steps=1000, generator_workers=1, generator_workers_eval=1, grad_clip=200, grad_clip_ac=200, gru_layers=1, gru_type='gru', hidden_dim=1000, imag_horizon=15, image_categorical=False, image_channels=3, image_decoder='cnn', image_decoder_layers=0, image_decoder_min_prob=0, image_encoder='cnn', image_encoder_layers=0, image_key='image', image_size=64, image_weight=1.0, iwae_samples=1, keep_state=True, kl_balance=0.8, kl_weight=1.0, lambda_gae=0.95, layer_norm=True, limit_step_ratio=0, log_interval=100, logbatch_interval=1000, map_categorical=True, map_channels=4, map_decoder='dense', map_hidden_dim=1024, map_hidden_layers=4, map_key=None, map_model='none', map_size=11, map_stoch_dim=64, mem_loss_type=None, mem_model='none', model='dreamer', n_env_steps=200000000, n_steps=99000000, offline_data_dir=None, offline_eval_dir=None, offline_prefill_dir=None, offline_test_dir=None, reset_interval=200, resume_id=None, reward_decoder_categorical=None, reward_decoder_layers=4, reward_input=False, reward_weight=1.0, run_name=None, save_interval=500, stoch_dim=32, stoch_discrete=32, target_interval=100, terminal_decoder_layers=4, terminal_weight=1.0, test_batch_size=10, test_batches=61, vecobs_weight=1.0, verbose=False)


obs = {}
# obs.keys() = (['reset', 'action', 'reward', 'image', 'mission', 'terminal', 'map', 'map_seen_mask', 'map_coord', 'vecobs'])
# actually_used = ["action", "reset", "terminal", "image", "vecobs", "reward"]
# action is discrete onehot (T, B, 3)  [0 1 0] 
# if obs terminal is 0 0 0 1 0 then obs reset is 0 0 0 0 1 (T, B)
# image is 0-1, float16, (T, B, C, H, W)
# vecobs is float, robotstate (T, B, 5)
# reward is float, (T, B)

iwae_samples = 1 # always 1 for training
# do_image_pred seems to be for logging only (nograd). in training loop:
# do_image_pred=steps % conf.log_interval >= int(conf.log_interval * 0.9)  # 10% of batches
do_image_pred = False
do_open_loop = False # always closed loop for training. open loop for eval

# in_state: Tuple[Tensor, Tensor],    # [(BI,D) (BI,S)] -> h, z
state = wm.init_state(conf.batch_size * conf.iwae_samples)


# called in training loop
for i in range(10):
    loss_model, features, states, out_state, metrics, tensors = \
        wm.training_step(obs,
                         in_state,
                         iwae_samples=iwae_samples,
                         do_open_loop=do_open_loop,
                         do_image_pred=do_image_pred)
state = out_state


# in eval
iwae_samples = 10
do_open_loop = True
# ... many variations, TODO clarify
