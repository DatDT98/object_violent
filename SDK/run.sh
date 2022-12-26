if ! hash python3; then
    echo "python3 is not installed. Please install python 3 version"
    exit 1
fi

ver=$(python3 -V 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1\2/')
if [ "$ver" > "27" ]; then
    echo "Run application"
        python3 test.py
    exit 1
else
        echo "This script requires python 3 or greater"
fi