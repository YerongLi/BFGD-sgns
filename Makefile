all:
	python3 setup.py build_ext --inplace
	rm -rf build

clean:
	rm -rf util/*.so util/*.pyc util/*.c \
		util/__pycache__
