# Robust Random Cut Forest for Streaming Anomaly Detection

This repository implements a method for streaming anomaly detection that was proposed here: 
http://proceedings.mlr.press/v48/guha16.pdf. The method is implemented in this public repository: https://github.com/kLabUM/rrc and New Knowledge's edited implementation is available here: https://github.com/NewKnowledge/rrcf. This approach maintains a forest of trees in which a data point's depth is inversely proportional to its anomaly score (points closer to the root of the tree = more anomalous). The anomaly score of a new data point is proportional to the change in complexity that results from inserting the new data point into each tree in the forest (averaged).

For the purposes of TA1 classification in the ASED program, this repository trains separate  anomaly detection "forests" on each individual account that's being monitored. When a new message is sent to the classifier, each individual account that receives the message produces a unique anomaly score. These anomaly scores are than averaged based on the number of points seen by each forest to produce an overall anomaly score. 

Finally, during the training phase, the algorithm fits the model for each account on batches of training data (messages). Then, during the testing phase, the algorithm both produces classificatin predictions and also updates each model according to an online approach. 

## gRPC Dockerized Classifierfor Deployment

1) `grapevine_pb2.py` and `grapevine_pb2_grpc.py` generated from `grapevine.proto` in `protos/` according to instructions in `protos/README.md`. These files must be generated every time `grapevine.proto` is changed
2) `rrcf_streaming_server.py`: a server that runs batch training jobs intermittently during the training phase and then produces predictions and streaming online updates during the testing phase (for now, all model states stored in server memory)
3) `rrcf_stream_only_server.py` a server that produces only streaming online updates during both the training and testing phase (for now, all model states stored in server memory)
4) `dry_run_test_client.py` an example test client that streams the dry run data sorted by timestamp through either of the streaming anomaly detection servers and evaluates the classification results. 
 
To build the corresponding docker image:
`docker build -t <image_name>:<version> .`

To run the docker image (can change port mapping):
`docker run -it -p 50050:50050 <image_name>:<version>`


