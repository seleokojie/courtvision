FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Fix expired Yarn key failure during apt-get update
# The base image has an expired key for Yarn. We remove the list file to prevent apt-get update failures
# during feature installation (like docker-in-docker).
RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && apt-get update
