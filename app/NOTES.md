# Notes

### Connect PGAdmin to Postgres

1. Pull postgres image from Docker Hub `docker pull postgres:latest`
2. Run the container using the below command `docker run -p 5432:5432 postgres`
3. Using docker's inspect command find the IP
4. Use that IP, PORT, Username, and Password to connect in PGADMIN specified in `.env`
5. You can also do a simple telnet like below to confirm if you can access docker postgres container: con`telnet IP_ADDRESS 5432`