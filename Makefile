all:
	python setup.py build_ext --inplace
	rm -rf build

visual:
	python visual_setup.py build_ext --inplace
clean:
	rm -rf util/*.so util/*.pyc util/*.c \
		util/__pycache__
