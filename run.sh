
**Link Prediction**
python main.py --dataset collab  --device_id 0
python main.py --dataset act --device_id 1
python main.py --dataset synthetic --P 0.4 --device_id 1
python main.py --dataset synthetic --P 0.6 --device_id 1
python main.py --dataset synthetic --P 0.8 --device_id 1

**Node Classification**
python main.py --dataset Aminer --device_id 0
python main.py --dataset dymotif_data --device_id 0