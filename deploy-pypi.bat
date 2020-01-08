IF EXIST dist del dist
python setup.py sdist bdist_wheel
twine upload dist/*