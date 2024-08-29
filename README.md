1. $ wget https://github.com/milvus-io/milvus/releases/download/v2.4.1/milvus-standalone-docker-compose.yml -O docker-compose.yml


2. $ docker compose up -d --> {start continer}
    
4. $ docker pull milvusdb/milvus-insight:latest  (optional) ----> {for milvus GUI }

5. $ docker run -d --name milvus-insight -p 8000:3000 milvusdb/milvus-insight:latest {}





----------------------------------------------------------------------------------------------------------------------------------------------


## reference 

1. https://medium.com/@samsushruth/installation-of-milvus-database-into-our-local-system-on-windows-operating-system-356879499e90
    a. [text](https://github.com/zilliztech/attu/releases/tag/v2.4.7)

2. [If error with fbgemmdll](https://www.reddit.com/r/learnpython/comments/1erj4hx/pytorch_dependency_error_fbgemmdll/?rdt=65521)

