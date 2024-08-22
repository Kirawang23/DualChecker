while true; do
    python3 main.py
    if [ $? -ne 0 ]; then
        echo "The script stopped, retrying in 1 minute..."
        sleep 60
    else
        echo "The script completed successfully."
        break
    fi
done