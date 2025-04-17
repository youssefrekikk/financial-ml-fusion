
```
financial-ml-fusion
├─ 0.13.0'`
├─ config.py
├─ data
│  ├─ AAPL.csv
│  ├─ AMZN.csv
│  ├─ BAC.csv
│  ├─ DIA.csv
│  ├─ DJI.csv
│  ├─ GOOGL.csv
│  ├─ IWM.csv
│  ├─ IXIC.csv
│  ├─ JPM.csv
│  ├─ META.csv
│  ├─ MSFT.csv
│  ├─ NVDA.csv
│  ├─ processed
│  ├─ QQQ.csv
│  ├─ raw
│  │  └─ traindata_gen.py
│  └─ TSLA.csv
├─ data_backup
│  ├─ AAPL_stock_data.csv
│  ├─ AMZN_stock_data.csv
│  ├─ economic_data.csv
│  ├─ GOOGL_stock_data.csv
│  ├─ JNJ_stock_data.csv
│  ├─ JPM_stock_data.csv
│  ├─ MSFT_stock_data.csv
│  ├─ NVDA_stock_data.csv
│  ├─ PG_stock_data.csv
│  ├─ TSLA_stock_data.csv
│  └─ V_stock_data.csv
├─ detailed_results
│  ├─ AAPL_tft_metrics.csv
│  ├─ AAPL_tft_model.ckpt
│  ├─ AAPL_tft_predictions.csv
│  ├─ AAPL_tft_predictions.png
│  ├─ AAPL_tft_signal_distribution.png
│  ├─ AMZN_tft_metrics.csv
│  ├─ AMZN_tft_model.ckpt
│  ├─ AMZN_tft_predictions.csv
│  ├─ AMZN_tft_predictions.png
│  ├─ AMZN_tft_signal_distribution.png
│  ├─ GOOGL_tft_metrics.csv
│  ├─ GOOGL_tft_model.ckpt
│  ├─ GOOGL_tft_predictions.csv
│  ├─ GOOGL_tft_predictions.png
│  ├─ GOOGL_tft_signal_distribution.png
│  ├─ META_tft_metrics.csv
│  ├─ META_tft_model.ckpt
│  ├─ META_tft_predictions.csv
│  ├─ META_tft_predictions.png
│  ├─ META_tft_signal_distribution.png
│  ├─ MSFT_tft_metrics.csv
│  ├─ MSFT_tft_model.ckpt
│  ├─ MSFT_tft_predictions.csv
│  ├─ MSFT_tft_predictions.png
│  └─ MSFT_tft_signal_distribution.png
├─ grid_search_results
│  ├─ AAPL_seq10_pred1_thresh0.001_20250402_233320.json
│  ├─ AAPL_seq10_pred1_thresh0.001_20250403_230128.json
│  ├─ AAPL_seq10_pred1_thresh0.002_20250402_233320.json
│  ├─ AAPL_seq10_pred1_thresh0.002_20250403_230128.json
│  ├─ AAPL_seq10_pred1_thresh0.005_20250402_233320.json
│  ├─ AAPL_seq10_pred1_thresh0.005_20250403_230128.json
│  ├─ AAPL_seq20_pred1_thresh0.001_20250402_233320.json
│  ├─ AAPL_seq20_pred1_thresh0.001_20250403_230128.json
│  ├─ AAPL_seq20_pred1_thresh0.002_20250402_233320.json
│  ├─ AAPL_seq20_pred1_thresh0.002_20250403_230128.json
│  ├─ AAPL_seq20_pred1_thresh0.005_20250402_233320.json
│  ├─ AAPL_seq20_pred1_thresh0.005_20250403_230128.json
│  ├─ AAPL_seq30_pred1_thresh0.001_20250402_233320.json
│  ├─ AAPL_seq30_pred1_thresh0.001_20250403_230128.json
│  ├─ AAPL_seq30_pred1_thresh0.002_20250402_233320.json
│  ├─ AAPL_seq30_pred1_thresh0.002_20250403_230128.json
│  ├─ AAPL_seq30_pred1_thresh0.005_20250402_233320.json
│  ├─ AAPL_seq30_pred1_thresh0.005_20250403_230128.json
│  ├─ AAPL_seq60_pred1_thresh0.001_20250402_233320.json
│  ├─ AAPL_seq60_pred1_thresh0.001_20250403_230128.json
│  ├─ AAPL_seq60_pred1_thresh0.002_20250402_233320.json
│  ├─ AAPL_seq60_pred1_thresh0.002_20250403_230128.json
│  ├─ AAPL_seq60_pred1_thresh0.005_20250402_233320.json
│  ├─ AAPL_seq60_pred1_thresh0.005_20250403_230128.json
│  ├─ AMZN_seq10_pred1_thresh0.001_20250403_230128.json
│  ├─ AMZN_seq10_pred1_thresh0.002_20250403_230128.json
│  ├─ AMZN_seq10_pred1_thresh0.005_20250403_230128.json
│  ├─ AMZN_seq20_pred1_thresh0.001_20250403_230128.json
│  ├─ AMZN_seq20_pred1_thresh0.002_20250403_230128.json
│  ├─ AMZN_seq20_pred1_thresh0.005_20250403_230128.json
│  ├─ AMZN_seq30_pred1_thresh0.001_20250403_230128.json
│  ├─ AMZN_seq30_pred1_thresh0.002_20250403_230128.json
│  ├─ AMZN_seq30_pred1_thresh0.005_20250403_230128.json
│  ├─ AMZN_seq60_pred1_thresh0.001_20250403_230128.json
│  ├─ AMZN_seq60_pred1_thresh0.002_20250403_230128.json
│  ├─ AMZN_seq60_pred1_thresh0.005_20250403_230128.json
│  ├─ best_parameters.json
│  ├─ GOOGL_seq10_pred1_thresh0.001_20250403_230128.json
│  ├─ GOOGL_seq10_pred1_thresh0.002_20250403_230128.json
│  ├─ GOOGL_seq10_pred1_thresh0.005_20250403_230128.json
│  ├─ GOOGL_seq20_pred1_thresh0.001_20250403_230128.json
│  ├─ GOOGL_seq20_pred1_thresh0.002_20250403_230128.json
│  ├─ GOOGL_seq20_pred1_thresh0.005_20250403_230128.json
│  ├─ GOOGL_seq30_pred1_thresh0.001_20250403_230128.json
│  ├─ GOOGL_seq30_pred1_thresh0.002_20250403_230128.json
│  ├─ GOOGL_seq30_pred1_thresh0.005_20250403_230128.json
│  ├─ GOOGL_seq60_pred1_thresh0.001_20250403_230128.json
│  ├─ GOOGL_seq60_pred1_thresh0.002_20250403_230128.json
│  ├─ GOOGL_seq60_pred1_thresh0.005_20250403_230128.json
│  ├─ grid_search_20250323_164558.json
│  ├─ grid_search_20250323_170336.json
│  ├─ grid_search_20250323_175344.json
│  ├─ grid_search_results_20250403_230128.csv
│  ├─ META_seq10_pred1_thresh0.001_20250403_230128.json
│  ├─ META_seq10_pred1_thresh0.002_20250403_230128.json
│  ├─ META_seq10_pred1_thresh0.005_20250403_230128.json
│  ├─ META_seq20_pred1_thresh0.001_20250403_230128.json
│  ├─ META_seq20_pred1_thresh0.002_20250403_230128.json
│  ├─ META_seq20_pred1_thresh0.005_20250403_230128.json
│  ├─ META_seq30_pred1_thresh0.001_20250403_230128.json
│  ├─ META_seq30_pred1_thresh0.002_20250403_230128.json
│  ├─ META_seq30_pred1_thresh0.005_20250403_230128.json
│  ├─ META_seq60_pred1_thresh0.001_20250403_230128.json
│  ├─ META_seq60_pred1_thresh0.002_20250403_230128.json
│  ├─ META_seq60_pred1_thresh0.005_20250403_230128.json
│  ├─ MSFT_seq10_pred1_thresh0.001_20250402_233320.json
│  ├─ MSFT_seq10_pred1_thresh0.001_20250403_230128.json
│  ├─ MSFT_seq10_pred1_thresh0.002_20250402_233320.json
│  ├─ MSFT_seq10_pred1_thresh0.002_20250403_230128.json
│  ├─ MSFT_seq10_pred1_thresh0.005_20250402_233320.json
│  ├─ MSFT_seq10_pred1_thresh0.005_20250403_230128.json
│  ├─ MSFT_seq20_pred1_thresh0.001_20250402_233320.json
│  ├─ MSFT_seq20_pred1_thresh0.001_20250403_230128.json
│  ├─ MSFT_seq20_pred1_thresh0.002_20250402_233320.json
│  ├─ MSFT_seq20_pred1_thresh0.002_20250403_230128.json
│  ├─ MSFT_seq20_pred1_thresh0.005_20250402_233320.json
│  ├─ MSFT_seq20_pred1_thresh0.005_20250403_230128.json
│  ├─ MSFT_seq30_pred1_thresh0.001_20250402_233320.json
│  ├─ MSFT_seq30_pred1_thresh0.001_20250403_230128.json
│  ├─ MSFT_seq30_pred1_thresh0.002_20250402_233320.json
│  ├─ MSFT_seq30_pred1_thresh0.002_20250403_230128.json
│  ├─ MSFT_seq30_pred1_thresh0.005_20250402_233320.json
│  ├─ MSFT_seq30_pred1_thresh0.005_20250403_230128.json
│  ├─ MSFT_seq60_pred1_thresh0.001_20250402_233320.json
│  ├─ MSFT_seq60_pred1_thresh0.001_20250403_230128.json
│  ├─ MSFT_seq60_pred1_thresh0.002_20250402_233320.json
│  ├─ MSFT_seq60_pred1_thresh0.002_20250403_230128.json
│  ├─ MSFT_seq60_pred1_thresh0.005_20250402_233320.json
│  └─ MSFT_seq60_pred1_thresh0.005_20250403_230128.json
├─ lightning_logs
│  ├─ lightning_logs
│  │  ├─ version_0
│  │  │  ├─ events.out.tfevents.1743112490.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_1
│  │  │  ├─ events.out.tfevents.1743113488.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_10
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=7-step=696.ckpt
│  │  │  ├─ events.out.tfevents.1743224808.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_100
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=7-step=552.ckpt
│  │  │  ├─ events.out.tfevents.1743725991.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_101
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=9-step=430.ckpt
│  │  │  ├─ events.out.tfevents.1743726126.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_102
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=5-step=192.ckpt
│  │  │  ├─ events.out.tfevents.1743726233.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_103
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=10-step=220.ckpt
│  │  │  ├─ events.out.tfevents.1743726282.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_104
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=11-step=732.ckpt
│  │  │  ├─ events.out.tfevents.1743726339.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_11
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=6-step=609.ckpt
│  │  │  ├─ events.out.tfevents.1743225421.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_12
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=5-step=522.ckpt
│  │  │  ├─ events.out.tfevents.1743225737.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_13
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=5-step=522.ckpt
│  │  │  ├─ events.out.tfevents.1743226628.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_14
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=6-step=609.ckpt
│  │  │  ├─ events.out.tfevents.1743262284.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_15
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=5-step=522.ckpt
│  │  │  ├─ events.out.tfevents.1743262658.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_16
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=8-step=783.ckpt
│  │  │  ├─ events.out.tfevents.1743618848.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_17
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=8-step=621.ckpt
│  │  │  ├─ events.out.tfevents.1743633201.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_18
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=6-step=483.ckpt
│  │  │  ├─ events.out.tfevents.1743633344.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_19
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=9-step=690.ckpt
│  │  │  ├─ events.out.tfevents.1743633462.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_2
│  │  │  ├─ events.out.tfevents.1743113559.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_20
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=6-step=483.ckpt
│  │  │  ├─ events.out.tfevents.1743633646.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_21
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=5-step=414.ckpt
│  │  │  ├─ events.out.tfevents.1743633782.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_22
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=7-step=552.ckpt
│  │  │  ├─ events.out.tfevents.1743633905.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_23
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=8-step=621.ckpt
│  │  │  ├─ events.out.tfevents.1743634059.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_24
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=14-step=1035.ckpt
│  │  │  ├─ events.out.tfevents.1743634259.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_25
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=7-step=552.ckpt
│  │  │  ├─ events.out.tfevents.1743634585.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_26
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=7-step=552.ckpt
│  │  │  ├─ events.out.tfevents.1743634776.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_27
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=6-step=483.ckpt
│  │  │  ├─ events.out.tfevents.1743635029.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_28
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=8-step=621.ckpt
│  │  │  ├─ events.out.tfevents.1743635251.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_29
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=10-step=671.ckpt
│  │  │  ├─ events.out.tfevents.1743635543.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_3
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=5-step=522.ckpt
│  │  │  ├─ events.out.tfevents.1743190601.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_30
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=8-step=549.ckpt
│  │  │  ├─ events.out.tfevents.1743635735.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_31
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=5-step=366.ckpt
│  │  │  ├─ events.out.tfevents.1743635883.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_32
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=11-step=732.ckpt
│  │  │  ├─ events.out.tfevents.1743635974.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_33
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=6-step=427.ckpt
│  │  │  ├─ events.out.tfevents.1743636172.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_34
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=11-step=732.ckpt
│  │  │  ├─ events.out.tfevents.1743636289.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_35
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=6-step=427.ckpt
│  │  │  ├─ events.out.tfevents.1743636501.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_36
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=7-step=488.ckpt
│  │  │  ├─ events.out.tfevents.1743636650.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_37
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=8-step=549.ckpt
│  │  │  ├─ events.out.tfevents.1743636824.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_38
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=8-step=540.ckpt
│  │  │  ├─ events.out.tfevents.1743637015.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_39
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=6-step=420.ckpt
│  │  │  ├─ events.out.tfevents.1743637280.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_4
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=8-step=783.ckpt
│  │  │  ├─ events.out.tfevents.1743191045.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_40
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=8-step=621.ckpt
│  │  │  ├─ events.out.tfevents.1743717688.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_41
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=9-step=690.ckpt
│  │  │  ├─ events.out.tfevents.1743717818.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_42
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=14-step=1035.ckpt
│  │  │  ├─ events.out.tfevents.1743717980.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_43
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=8-step=621.ckpt
│  │  │  ├─ events.out.tfevents.1743718227.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_44
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=16-step=1173.ckpt
│  │  │  ├─ events.out.tfevents.1743718395.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_45
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=6-step=483.ckpt
│  │  │  ├─ events.out.tfevents.1743718712.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_46
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=10-step=759.ckpt
│  │  │  ├─ events.out.tfevents.1743718848.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_47
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=6-step=483.ckpt
│  │  │  ├─ events.out.tfevents.1743719085.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_48
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=5-step=414.ckpt
│  │  │  ├─ events.out.tfevents.1743719238.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_49
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=8-step=621.ckpt
│  │  │  ├─ events.out.tfevents.1743719371.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_5
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=9-step=870.ckpt
│  │  │  ├─ events.out.tfevents.1743191792.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_50
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=11-step=828.ckpt
│  │  │  ├─ events.out.tfevents.1743719650.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_51
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=5-step=414.ckpt
│  │  │  ├─ events.out.tfevents.1743720018.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_52
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=6-step=427.ckpt
│  │  │  ├─ events.out.tfevents.1743720208.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_53
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=11-step=732.ckpt
│  │  │  ├─ events.out.tfevents.1743720313.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_54
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=6-step=427.ckpt
│  │  │  ├─ events.out.tfevents.1743720490.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_55
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=6-step=427.ckpt
│  │  │  ├─ events.out.tfevents.1743720597.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_56
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=5-step=366.ckpt
│  │  │  ├─ events.out.tfevents.1743720717.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_57
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=5-step=366.ckpt
│  │  │  ├─ events.out.tfevents.1743720820.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_58
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=13-step=854.ckpt
│  │  │  ├─ events.out.tfevents.1743720925.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_59
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=9-step=610.ckpt
│  │  │  ├─ events.out.tfevents.1743721194.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_6
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=5-step=522.ckpt
│  │  │  ├─ events.out.tfevents.1743192332.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_60
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=19-step=1220.ckpt
│  │  │  ├─ events.out.tfevents.1743721387.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_61
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=16-step=1020.ckpt
│  │  │  ├─ events.out.tfevents.1743721778.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_62
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=7-step=480.ckpt
│  │  │  ├─ events.out.tfevents.1743722230.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_63
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=5-step=360.ckpt
│  │  │  ├─ events.out.tfevents.1743722446.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_64
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=6-step=224.ckpt
│  │  │  ├─ events.out.tfevents.1743722612.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_65
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=10-step=352.ckpt
│  │  │  ├─ events.out.tfevents.1743722670.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_66
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=12-step=416.ckpt
│  │  │  ├─ events.out.tfevents.1743722757.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_67
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=6-step=224.ckpt
│  │  │  ├─ events.out.tfevents.1743722862.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_68
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=8-step=288.ckpt
│  │  │  ├─ events.out.tfevents.1743722926.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_69
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=6-step=224.ckpt
│  │  │  ├─ events.out.tfevents.1743723008.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_7
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=6-step=609.ckpt
│  │  │  ├─ events.out.tfevents.1743192587.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_70
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=10-step=352.ckpt
│  │  │  ├─ events.out.tfevents.1743723073.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_71
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=6-step=224.ckpt
│  │  │  ├─ events.out.tfevents.1743723187.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_72
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=6-step=224.ckpt
│  │  │  ├─ events.out.tfevents.1743723260.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_73
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=7-step=248.ckpt
│  │  │  ├─ events.out.tfevents.1743723334.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_74
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=6-step=217.ckpt
│  │  │  ├─ events.out.tfevents.1743723449.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_75
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=8-step=279.ckpt
│  │  │  ├─ events.out.tfevents.1743723549.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_76
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=7-step=344.ckpt
│  │  │  ├─ events.out.tfevents.1743723678.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_77
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=10-step=473.ckpt
│  │  │  ├─ events.out.tfevents.1743723764.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_78
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=7-step=344.ckpt
│  │  │  ├─ events.out.tfevents.1743723880.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_79
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=10-step=473.ckpt
│  │  │  ├─ events.out.tfevents.1743723965.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_8
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=6-step=609.ckpt
│  │  │  ├─ events.out.tfevents.1743192792.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_80
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=8-step=387.ckpt
│  │  │  ├─ events.out.tfevents.1743724096.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_81
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=10-step=473.ckpt
│  │  │  ├─ events.out.tfevents.1743724204.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_82
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=6-step=301.ckpt
│  │  │  ├─ events.out.tfevents.1743724339.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_83
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=9-step=430.ckpt
│  │  │  ├─ events.out.tfevents.1743724437.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_84
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=7-step=344.ckpt
│  │  │  ├─ events.out.tfevents.1743724574.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_85
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=9-step=430.ckpt
│  │  │  ├─ events.out.tfevents.1743724687.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_86
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=11-step=516.ckpt
│  │  │  ├─ events.out.tfevents.1743724882.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_87
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=8-step=387.ckpt
│  │  │  ├─ events.out.tfevents.1743725114.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_88
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=11-step=240.ckpt
│  │  │  ├─ events.out.tfevents.1743725291.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_89
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=8-step=180.ckpt
│  │  │  ├─ events.out.tfevents.1743725354.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_9
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=9-step=870.ckpt
│  │  │  ├─ events.out.tfevents.1743194194.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_90
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=6-step=140.ckpt
│  │  │  ├─ events.out.tfevents.1743725401.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_91
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=7-step=160.ckpt
│  │  │  ├─ events.out.tfevents.1743725438.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_92
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=7-step=152.ckpt
│  │  │  ├─ events.out.tfevents.1743725485.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_93
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=12-step=247.ckpt
│  │  │  ├─ events.out.tfevents.1743725530.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_94
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=8-step=171.ckpt
│  │  │  ├─ events.out.tfevents.1743725603.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_95
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=8-step=171.ckpt
│  │  │  ├─ events.out.tfevents.1743725661.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_96
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=7-step=152.ckpt
│  │  │  ├─ events.out.tfevents.1743725718.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_97
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=7-step=152.ckpt
│  │  │  ├─ events.out.tfevents.1743725769.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  ├─ version_98
│  │  │  ├─ checkpoints
│  │  │  │  └─ epoch=8-step=171.ckpt
│  │  │  ├─ events.out.tfevents.1743725841.DESKTOP-F46T2PB
│  │  │  └─ hparams.yaml
│  │  └─ version_99
│  │     ├─ checkpoints
│  │     │  └─ epoch=6-step=133.ckpt
│  │     ├─ events.out.tfevents.1743725921.DESKTOP-F46T2PB
│  │     └─ hparams.yaml
│  ├─ version_0
│  │  ├─ events.out.tfevents.1743190713.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_1
│  │  ├─ events.out.tfevents.1743191214.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_10
│  │  ├─ events.out.tfevents.1743224996.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_100
│  │  ├─ events.out.tfevents.1743722666.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_101
│  │  ├─ events.out.tfevents.1743722667.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_102
│  │  ├─ events.out.tfevents.1743722668.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_103
│  │  ├─ events.out.tfevents.1743722756.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_104
│  │  ├─ events.out.tfevents.1743722860.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_105
│  │  ├─ events.out.tfevents.1743722923.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_106
│  │  ├─ events.out.tfevents.1743722924.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_107
│  │  ├─ events.out.tfevents.1743722925.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_108
│  │  ├─ events.out.tfevents.1743723007.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_109
│  │  ├─ events.out.tfevents.1743723072.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_11
│  │  ├─ events.out.tfevents.1743224998.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_110
│  │  ├─ events.out.tfevents.1743723183.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_111
│  │  ├─ events.out.tfevents.1743723185.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_112
│  │  ├─ events.out.tfevents.1743723186.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_113
│  │  ├─ events.out.tfevents.1743723258.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_114
│  │  ├─ events.out.tfevents.1743723333.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_115
│  │  ├─ events.out.tfevents.1743723444.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_116
│  │  ├─ events.out.tfevents.1743723446.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_117
│  │  ├─ events.out.tfevents.1743723447.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_118
│  │  ├─ events.out.tfevents.1743723547.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_119
│  │  ├─ events.out.tfevents.1743723676.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_12
│  │  ├─ events.out.tfevents.1743225632.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_120
│  │  ├─ events.out.tfevents.1743723759.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_121
│  │  ├─ events.out.tfevents.1743723761.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_122
│  │  ├─ events.out.tfevents.1743723762.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_123
│  │  ├─ events.out.tfevents.1743723878.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_124
│  │  ├─ events.out.tfevents.1743723964.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_125
│  │  ├─ events.out.tfevents.1743724092.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_126
│  │  ├─ events.out.tfevents.1743724093.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_127
│  │  ├─ events.out.tfevents.1743724094.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_128
│  │  ├─ events.out.tfevents.1743724203.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_129
│  │  ├─ events.out.tfevents.1743724337.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_13
│  │  ├─ events.out.tfevents.1743225634.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_130
│  │  ├─ events.out.tfevents.1743724432.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_131
│  │  ├─ events.out.tfevents.1743724433.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_132
│  │  ├─ events.out.tfevents.1743724435.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_133
│  │  ├─ events.out.tfevents.1743724572.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_134
│  │  ├─ events.out.tfevents.1743724685.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_135
│  │  ├─ events.out.tfevents.1743724875.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_136
│  │  ├─ events.out.tfevents.1743724877.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_137
│  │  ├─ events.out.tfevents.1743724879.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_138
│  │  ├─ events.out.tfevents.1743725112.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_139
│  │  ├─ events.out.tfevents.1743725288.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_14
│  │  ├─ events.out.tfevents.1743225900.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_140
│  │  ├─ events.out.tfevents.1743725351.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_141
│  │  ├─ events.out.tfevents.1743725352.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_142
│  │  ├─ events.out.tfevents.1743725353.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_143
│  │  ├─ events.out.tfevents.1743725400.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_144
│  │  ├─ events.out.tfevents.1743725437.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_145
│  │  ├─ events.out.tfevents.1743725482.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_146
│  │  ├─ events.out.tfevents.1743725483.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_147
│  │  ├─ events.out.tfevents.1743725484.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_148
│  │  ├─ events.out.tfevents.1743725528.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_149
│  │  ├─ events.out.tfevents.1743725602.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_15
│  │  ├─ events.out.tfevents.1743225901.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_150
│  │  ├─ events.out.tfevents.1743725658.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_151
│  │  ├─ events.out.tfevents.1743725659.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_152
│  │  ├─ events.out.tfevents.1743725659.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_153
│  │  ├─ events.out.tfevents.1743725716.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_154
│  │  ├─ events.out.tfevents.1743725768.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_155
│  │  ├─ events.out.tfevents.1743725838.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_156
│  │  ├─ events.out.tfevents.1743725839.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_157
│  │  ├─ events.out.tfevents.1743725840.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_158
│  │  ├─ events.out.tfevents.1743725920.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_159
│  │  ├─ events.out.tfevents.1743725983.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_16
│  │  ├─ events.out.tfevents.1743226744.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_160
│  │  ├─ events.out.tfevents.1743726123.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_161
│  │  ├─ events.out.tfevents.1743726231.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_162
│  │  ├─ events.out.tfevents.1743726280.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_163
│  │  ├─ events.out.tfevents.1743726337.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_164
│  │  ├─ events.out.tfevents.1743726514.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_17
│  │  ├─ events.out.tfevents.1743226745.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_18
│  │  ├─ events.out.tfevents.1743262452.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_19
│  │  ├─ events.out.tfevents.1743262453.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_2
│  │  ├─ events.out.tfevents.1743191976.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_20
│  │  ├─ events.out.tfevents.1743262795.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_21
│  │  ├─ events.out.tfevents.1743619008.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_22
│  │  ├─ events.out.tfevents.1743633338.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_23
│  │  ├─ events.out.tfevents.1743633340.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_24
│  │  ├─ events.out.tfevents.1743633342.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_25
│  │  ├─ events.out.tfevents.1743633460.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_26
│  │  ├─ events.out.tfevents.1743633644.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_27
│  │  ├─ events.out.tfevents.1743633775.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_28
│  │  ├─ events.out.tfevents.1743633777.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_29
│  │  ├─ events.out.tfevents.1743633779.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_3
│  │  ├─ events.out.tfevents.1743192449.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_30
│  │  ├─ events.out.tfevents.1743633902.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_31
│  │  ├─ events.out.tfevents.1743634057.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_32
│  │  ├─ events.out.tfevents.1743634252.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_33
│  │  ├─ events.out.tfevents.1743634254.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_34
│  │  ├─ events.out.tfevents.1743634257.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_35
│  │  ├─ events.out.tfevents.1743634583.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_36
│  │  ├─ events.out.tfevents.1743634773.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_37
│  │  ├─ events.out.tfevents.1743635019.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_38
│  │  ├─ events.out.tfevents.1743635022.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_39
│  │  ├─ events.out.tfevents.1743635025.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_4
│  │  ├─ events.out.tfevents.1743192450.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_40
│  │  ├─ events.out.tfevents.1743635247.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_41
│  │  ├─ events.out.tfevents.1743635539.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_42
│  │  ├─ events.out.tfevents.1743635727.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_43
│  │  ├─ events.out.tfevents.1743635730.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_44
│  │  ├─ events.out.tfevents.1743635732.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_45
│  │  ├─ events.out.tfevents.1743635881.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_46
│  │  ├─ events.out.tfevents.1743635972.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_47
│  │  ├─ events.out.tfevents.1743636166.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_48
│  │  ├─ events.out.tfevents.1743636168.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_49
│  │  ├─ events.out.tfevents.1743636170.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_5
│  │  ├─ events.out.tfevents.1743192763.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_50
│  │  ├─ events.out.tfevents.1743636287.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_51
│  │  ├─ events.out.tfevents.1743636499.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_52
│  │  ├─ events.out.tfevents.1743636643.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_53
│  │  ├─ events.out.tfevents.1743636646.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_54
│  │  ├─ events.out.tfevents.1743636648.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_55
│  │  ├─ events.out.tfevents.1743636821.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_56
│  │  ├─ events.out.tfevents.1743637012.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_57
│  │  ├─ events.out.tfevents.1743637272.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_58
│  │  ├─ events.out.tfevents.1743637274.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_59
│  │  ├─ events.out.tfevents.1743637277.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_6
│  │  ├─ events.out.tfevents.1743192764.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_60
│  │  ├─ events.out.tfevents.1743717812.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_61
│  │  ├─ events.out.tfevents.1743717814.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_62
│  │  ├─ events.out.tfevents.1743717816.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_63
│  │  ├─ events.out.tfevents.1743717978.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_64
│  │  ├─ events.out.tfevents.1743718224.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_65
│  │  ├─ events.out.tfevents.1743718389.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_66
│  │  ├─ events.out.tfevents.1743718391.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_67
│  │  ├─ events.out.tfevents.1743718393.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_68
│  │  ├─ events.out.tfevents.1743718710.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_69
│  │  ├─ events.out.tfevents.1743718846.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_7
│  │  ├─ events.out.tfevents.1743192950.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_70
│  │  ├─ events.out.tfevents.1743719078.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_71
│  │  ├─ events.out.tfevents.1743719080.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_72
│  │  ├─ events.out.tfevents.1743719082.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_73
│  │  ├─ events.out.tfevents.1743719235.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_74
│  │  ├─ events.out.tfevents.1743719368.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_75
│  │  ├─ events.out.tfevents.1743719640.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_76
│  │  ├─ events.out.tfevents.1743719643.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_77
│  │  ├─ events.out.tfevents.1743719646.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_78
│  │  ├─ events.out.tfevents.1743720015.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_79
│  │  ├─ events.out.tfevents.1743720204.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_8
│  │  ├─ events.out.tfevents.1743194406.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_80
│  │  ├─ events.out.tfevents.1743720307.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_81
│  │  ├─ events.out.tfevents.1743720309.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_82
│  │  ├─ events.out.tfevents.1743720311.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_83
│  │  ├─ events.out.tfevents.1743720487.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_84
│  │  ├─ events.out.tfevents.1743720594.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_85
│  │  ├─ events.out.tfevents.1743720711.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_86
│  │  ├─ events.out.tfevents.1743720713.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_87
│  │  ├─ events.out.tfevents.1743720715.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_88
│  │  ├─ events.out.tfevents.1743720817.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_89
│  │  ├─ events.out.tfevents.1743720922.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_9
│  │  ├─ events.out.tfevents.1743194407.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_90
│  │  ├─ events.out.tfevents.1743721187.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_91
│  │  ├─ events.out.tfevents.1743721189.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_92
│  │  ├─ events.out.tfevents.1743721192.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_93
│  │  ├─ events.out.tfevents.1743721384.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_94
│  │  ├─ events.out.tfevents.1743721775.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_95
│  │  ├─ events.out.tfevents.1743722221.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_96
│  │  ├─ events.out.tfevents.1743722224.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_97
│  │  ├─ events.out.tfevents.1743722226.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  ├─ version_98
│  │  ├─ events.out.tfevents.1743722443.DESKTOP-F46T2PB
│  │  └─ hparams.yaml
│  └─ version_99
│     ├─ events.out.tfevents.1743722609.DESKTOP-F46T2PB
│     └─ hparams.yaml
├─ logs
│  └─ transformer_log.txt
├─ main.py
├─ models
│  ├─ AAPL_model.pth
│  ├─ AAPL_tft_model.ckpt
│  ├─ AAPL_transformer_model.pt
│  └─ transformer_model.pt
├─ notebooks
├─ plots
│  ├─ AAPL_Direction_Accuracy_heatmap_thresh0.001.png
│  ├─ AAPL_Direction_Accuracy_heatmap_thresh0.002.png
│  ├─ AAPL_Direction_Accuracy_heatmap_thresh0.005.png
│  ├─ AAPL_sharpe_ratio_heatmap_thresh0.001.png
│  ├─ AAPL_sharpe_ratio_heatmap_thresh0.002.png
│  ├─ AAPL_sharpe_ratio_heatmap_thresh0.005.png
│  ├─ AMZN_Direction_Accuracy_heatmap_thresh0.001.png
│  ├─ AMZN_Direction_Accuracy_heatmap_thresh0.002.png
│  ├─ AMZN_Direction_Accuracy_heatmap_thresh0.005.png
│  ├─ AMZN_sharpe_ratio_heatmap_thresh0.001.png
│  ├─ AMZN_sharpe_ratio_heatmap_thresh0.002.png
│  ├─ AMZN_sharpe_ratio_heatmap_thresh0.005.png
│  ├─ GOOGL_Direction_Accuracy_heatmap_thresh0.001.png
│  ├─ GOOGL_Direction_Accuracy_heatmap_thresh0.002.png
│  ├─ GOOGL_Direction_Accuracy_heatmap_thresh0.005.png
│  ├─ GOOGL_sharpe_ratio_heatmap_thresh0.001.png
│  ├─ GOOGL_sharpe_ratio_heatmap_thresh0.002.png
│  ├─ GOOGL_sharpe_ratio_heatmap_thresh0.005.png
│  ├─ META_Direction_Accuracy_heatmap_thresh0.001.png
│  ├─ META_Direction_Accuracy_heatmap_thresh0.002.png
│  ├─ META_Direction_Accuracy_heatmap_thresh0.005.png
│  ├─ META_sharpe_ratio_heatmap_thresh0.001.png
│  ├─ META_sharpe_ratio_heatmap_thresh0.002.png
│  ├─ META_sharpe_ratio_heatmap_thresh0.005.png
│  ├─ MSFT_Direction_Accuracy_heatmap_thresh0.001.png
│  ├─ MSFT_Direction_Accuracy_heatmap_thresh0.002.png
│  ├─ MSFT_Direction_Accuracy_heatmap_thresh0.005.png
│  ├─ MSFT_sharpe_ratio_heatmap_thresh0.001.png
│  ├─ MSFT_sharpe_ratio_heatmap_thresh0.002.png
│  └─ MSFT_sharpe_ratio_heatmap_thresh0.005.png
├─ python
├─ requirements.txt
├─ results
│  ├─ AAPL_predictions.csv
│  ├─ AAPL_tft_metrics.csv
│  ├─ AAPL_tft_predictions.csv
│  ├─ AMZN_predictions.csv
│  ├─ evaluation_summary.csv
│  ├─ GOOGL_predictions.csv
│  ├─ JPM_predictions.csv
│  ├─ META_predictions.csv
│  ├─ MSFT_predictions.csv
│  ├─ NVDA_predictions.csv
│  ├─ simulation_summary.csv
│  └─ TSLA_predictions.csv
└─ src
   ├─ download_historical_data.py
   ├─ ensemble
   ├─ hmm_exploration.ipynb
   ├─ improved_hmm_model.py
   ├─ legacy
   │  ├─ hmm_model.py
   │  ├─ test.py
   │  ├─ test_hmm.py
   │  ├─ test_transformer.py
   │  ├─ TFT2.py
   │  └─ time_series_transformer.py
   ├─ test_improved_hmm.py
   ├─ test_simulator.py
   ├─ test_TFT.py
   ├─ TFT_forcasting.py
   ├─ trade_simulator.py
   └─ utils
      ├─ data_loader.py
      ├─ feature_engineering.py
      ├─ monte_carlo.py
      ├─ permutation_tests.py
      └─ __init__.py

```