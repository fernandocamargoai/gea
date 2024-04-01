#!/bin/bash

apt update && apt upgrade -y
apt-get --allow-releaseinfo-change update
apt-get update -q && apt-get upgrade -y
