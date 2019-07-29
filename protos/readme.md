grapevine.proto specs here need to match those in the Recourse grapevine repo:

https://github.com/uncharted-recourse/grapevine/grapevine/grapevine.proto

TODO -- ideally, we should reference those protospecs directly.


NOTE: If `grapevine.proto` file is changed, the gRPC python code needs to be re-generated as follows:

From top directory, run:

`python -m grpc_tools.protoc -Iprotos --python_out=. --grpc_python_out=. protos/grapevine.proto`

This will regenerate the `grapevine.pb2.py` and `grapevine.pb2_grpc.py` files in the spam_clf_endpoint directory

