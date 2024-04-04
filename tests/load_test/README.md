# Running the load test

The load test is implemented as a Python script that uses [Locust](https://locust.io/) library. It can be run as a
standalone script or as a Docker container.

## Running as a standalone script

You can run the load test:

```bash
locust -f tests/load_test/locustfile.py
```