#! /bin/bash

pip install -r requirement.txt

sudo apt update
echo | sudo apt install -y postgresql-common
echo | sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh
echo | sudo apt install postgresql-15-pgvector
sudo service postgresql start

sudo -u postgres psql -c "CREATE ROLE admin WITH LOGIN PASSWORD 'admin';"
sudo -u postgres psql -c "ALTER ROLE admin SUPERUSER;"
sudo -u postgres psql -c "CREATE DATABASE rag_vector_db;"
sudo -u postgres psql -c "CREATE USER flock WITH LOGIN PASSWORD 'flock';"
sudo -u postgres psql -c "ALTER DATABASE rag_vector_db OWNER TO flock;"