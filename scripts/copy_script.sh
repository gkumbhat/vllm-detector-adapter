#!/bin/bash

# Copy files from a specific location to the desired destination
cp -r /app/target_packages/*  ${SHARED_PACKAGE_PATH}

# # Run the main command
# exec "$@"