#!/usr/bin/env bash

stubgen ./pyece -o .
python setup.py sdist bdist_wheel
twine upload --repository pypi dist/*
find pyece -name "*.pyi" -type f -delete
rm -r dist build
