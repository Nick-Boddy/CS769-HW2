# Step 0. Change this to your campus ID
CAMPUSID='9068748822'
mkdir -p $CAMPUSID

# Step 1. (Optional) Any preprocessing step, e.g., downloading pre-trained word embeddings


# Step 2. Train models on two datasets.
##  2.1. Run experiments on SST
PREF='sst'
python3 classifier.py \
    --use_gpu \
    --option finetune \
    --lr 1e-5 \
    --seed 1234 \
    --train "data/${PREF}-train.txt" \
    --dev "data/${PREF}-dev.txt" \
    --test "data/${PREF}-test.txt" \
    --dev_out "${CAMPUSID}/${PREF}-dev-output.txt" \
    --test_out "${CAMPUSID}/${PREF}-test-output.txt" \
    --batch_size 64 \
    --filepath "${CAMPUSID}/${PREF}-model.pt" | tee ${CAMPUSID}/${PREF}-train-log.txt
    

##  2.2 Run experiments on CF-IMDB
PREF='cfimdb'
python3 classifier.py \
    --use_gpu \
    --option finetune \
    --lr 1e-5 \
    --seed 1234 \
    --train "data/${PREF}-train.txt" \
    --dev "data/${PREF}-dev.txt" \
    --test "data/${PREF}-test.txt" \
    --dev_out "${CAMPUSID}/${PREF}-dev-output.txt" \
    --test_out "${CAMPUSID}/${PREF}-test-output.txt" \
    --batch_size 8 \
    --filepath "${CAMPUSID}/${PREF}-model.pt" | tee ${CAMPUSID}/${PREF}-train-log.txt
    

# Step 3. Prepare submission:
##  3.1. Copy your code to the $CAMPUSID folder
for file in *.py; do cp $file ${CAMPUSID}/; done
for file in *.sh; do cp $file ${CAMPUSID}/; done
for file in *.md; do cp $file ${CAMPUSID}/; done
for file in *.txt; do cp $file ${CAMPUSID}/; done
for file in *.data; do cp $file ${CAMPUSID}/; done

##  3.2. Compress the $CAMPUSID folder to $CAMPUSID.zip
python prepare_submit.py ${CAMPUSID} ${CAMPUSID}
##  3.3. Submit the zip file to Canvas! Congrats!
