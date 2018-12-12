python ~/mxnet/tools/launch.py -n 2 -s 2 -H hosts \
    --sync-dst-dir /home/ubuntu/cifar10_dist \
    --launcher ssh \
    "python /home/ubuntu/cifar10_dist/cifar10_dist.py"
