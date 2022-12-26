sudo cp -r lib-websocket-client/ /usr/lib/
#if ! hash python3; then
#    echo "python is not installed. Please install python 3 version"
#    exit 1
#fi
#if ! hash pip3; then
#    echo "pip3 is not installed. Please install pip 3 version"
#    exit 1
#fi
#ver=$(pip -V 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1\2/')
#if [ "$ver" > "27" ]; then
#        pip3 install numpy
#    echo "-------------Library is ok ok--------------"
#    exit 1
#else
#        echo "This script requires python 3 or greater"
#fi
#
#pip install