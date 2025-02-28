# 1 Create S3 Bucket

# 2 Download model to a bastion host and upload to S3
[ec2-user@ip-10-0-95-118 ~]$ sudo dnf -y install python3-pip git
[ec2-user@ip-10-0-95-118 ~]$ sudo dnf install -y awscli
[ec2-user@ip-10-0-95-118 ~]$ pip install --upgrade huggingface_hub
[ec2-user@ip-10-0-95-118 ~]$ git config --global credential.helper store
[ec2-user@ip-10-0-95-118 ~]$ huggingface-cli login --token $HF_TOKEN --add-to-git-credential
[ec2-user@ip-10-0-95-118 ~]$ huggingface-cli download tiiuae/falcon-40b-instruct
[ec2-user@ip-10-0-95-118 ~]$ aws configure
[ec2-user@ip-10-0-95-118 ~]$ aws s3 sync .cache/huggingface/hub/models--tiiuae--falcon-40b-instruct/ s3://rhoai-llm-model/falcon-40b-instruct/ 

# 3 Create a RWX PVC
-> pvc-falcon-40b.yaml

# 4 Download the model to PV
-> pod-download-model-to-pv.yaml

PS: In the near future, it will not be necessary to download the model to the PV, since distributed inference will be available directly through the RHOAI Dashboard, pointing directly to S3 compatible storage.

# 5 Create the vllm-multinode-runtime custom runtime:
oc process vllm-multinode-runtime-template -n redhat-ods-applications|oc apply -n kserve-demo -f -

# 6 Deploy the model using the following InferenceService configuration:
-> InferenceService.yaml

# 7 Validade the Deploy
wrosalem@Mac % oc get pods
NAME                                                   READY   STATUS      RESTARTS      AGE
download-falcon-40b                                    0/1     Completed   0             78m
falcon-40b-instruct-predictor-bcd85bdd4-tnq79          0/1     Running     1 (13s ago)   78s
falcon-40b-instruct-predictor-worker-8749fb9f4-29ppl   0/1     Running     1 (7s ago)    78s

wrosalem@Mac % oc get InferenceService
NAME                  URL                                                   READY   PREV   LATEST   PREVROLLEDOUTREVISION   LATESTREADYREVISION   AGE
falcon-40b-instruct   https://falcon-40b-instruct-kserve-demo.example.com   False                                                                 103s