sudo insmod program2.ko
sleep 2
sudo rmmod program2
sudo dmesg | tail -10
