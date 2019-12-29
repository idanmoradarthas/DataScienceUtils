IF EXIST dist del dist
python setup.py sdist bdist_wheel
twine upload --repository-url https://pypi.org/project/data-science-utils/ dist/*