#!/usr/bin/env bash
EXP_NAME=$1
python train.py $EXP_NAME \
--secret_size 100 \
--num_steps 140000 \
--no_im_loss_steps 1500 \
--rnd_trans_ramp 10000 \
--l2_loss_ramp 15000 \
--lpips_loss_ramp 15000  \
--G_loss_ramp 15000 \
--rnd_trans .1  \
--secret_loss_scale 1.5 \
--l2_loss_scale 2 \
--lpips_loss_scale 1.5 \
--G_loss_scale .5 \
--y_scale 1 \
--u_scale 100 \
--v_scale 100 \
--borders white \
--jpeg_quality 50 \
--l2_edge_gain 10 \
--l2_edge_ramp 10000 \
--l2_edge_delay 80000
