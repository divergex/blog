---
title: "Messaging Patterns with ZeroMQ"
date: 2025-03-09 10:00:00 +0530
tags:
  - messaging
  - zeromq
---

# Messaging Patterns - What are they and how to use them with ZeroMQ

Messaging systems are essential for building scalable and reliable distributed
applications.
They provide a way for different components and processes to communicate and
coordinate actions and data exchange.
Messaging systems are usually very simple and easy to use, but they can be very
powerful and deal with a multitude of complex scenarios.

Messaging patterns define the structure and behavior of message exchanges
between components in a distributed system,
specifically dealing with how messages are sent, received, and processed.

ZeroMQ (ØMQ) is a high-performance asynchronous messaging library that provides
several built-in messaging patterns for distributed systems. These patterns help
structure communication between processes, threads, or networked applications.
In this post, we'll explore the key messaging patterns in ZeroMQ,
each with a small code example to demonstrate how they work.

## Request-Reply (REQ/REP)

The Request-Reply pattern is a synchronous communication model where a client
sends a request and waits for a response.
It is a simple and reliable pattern that ensures messages are delivered and
processed in order.

### Example:

Server (REP Socket)

```python
import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

while True:
  message = socket.recv()
print(f"Received request: {message.decode()}")
socket.send(b"Hello from server")
```

Client (REQ Socket)

```python
import zmq

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

socket.send(b"Hello")
response = socket.recv()
print(f"Received reply: {response.decode()}")
```

Use Case:
Remote procedure calls (RPC) and service-oriented architectures (SOA).

## Publish-Subscribe (PUB/SUB)

The Publish-Subscribe pattern allows a publisher to broadcast messages to
multiple subscribers. Subscribers can filter messages based on topics.

### Example:

Publisher (PUB Socket)

```python
import zmq
import time

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5556")

time.sleep(1)  # Allow subscribers to connect

while True:
  socket.send_multipart([b"topic1", b"Hello Subscribers!"])
time.sleep(1)
```

Subscriber (SUB Socket)

```python
import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5556")
socket.setsockopt_string(zmq.SUBSCRIBE, "topic1")  # Subscribe to topic1

while True:
  topic, message = socket.recv_multipart()
print(f"Received on {topic.decode()}: {message.decode()}")
```

Use Case:

Live data feeds (market data, sensor networks) and other Event-driven systems

## Push-Pull (PUSH/PULL)

The Push-Pull pattern is used for load balancing where a PUSH socket distributes
messages among multiple PULL workers.

### Example:

Producer (PUSH Socket)

```python
import zmq
import time

context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.bind("tcp://*:5557")

for i in range(10):
  socket.send_string(f"Task {i}")
time.sleep(0.5)
```

Worker (PULL Socket)

```python
import zmq

context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.connect("tcp://localhost:5557")

while True:
  task = socket.recv_string()
print(f"Processing: {task}")
```

Use Case:

Task distribution among workers and similar types of Load balancing

## Dealer-Router (DEALER/ROUTER)

The Dealer-Router pattern extends the Request-Reply model by allowing multiple
clients and handling asynchronous messaging. It provides advanced routing
by allowing clients to send messages with an identity frame.

### Example:

Server (ROUTER Socket)

```python
import zmq

context = zmq.Context()
socket = context.socket(zmq.ROUTER)
socket.bind("tcp://*:5558")

while True:
  identity, message = socket.recv_multipart()
print(f"Received from {identity.decode()}: {message.decode()}")
socket.send_multipart([identity, b"Reply from server"])
```

Client (DEALER Socket)

```python
import zmq

context = zmq.Context()
socket = context.socket(zmq.DEALER)
socket.setsockopt_string(zmq.IDENTITY, "Client1")
socket.connect("tcp://localhost:5558")

socket.send(b"Hello Server")
response = socket.recv()
print(f"Received: {response.decode()}")
```

Use Case:

Asynchronous client-server interactions, including load balancing with
identity-based routing

## Pair (PAIR/PAIR) – Peer-to-Peer Communication

The Pair pattern enables direct peer-to-peer messaging between two endpoints.
Unlike other patterns, PAIR sockets can only connect to one other PAIR socket,
making them ideal for inter-thread communication.

### Example:

Peer 1 (PAIR Socket)

```python
import zmq
import time

context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind("tcp://*:5560")

# Give the second peer time to connect
time.sleep(1)

socket.send(b"Hello from Peer 1")
message = socket.recv()
print(f"Received: {message.decode()}")
```

Peer 2 (PAIR Socket)

```python
import zmq

context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://localhost:5560")

message = socket.recv()
print(f"Received: {message.decode()}")
socket.send(b"Hello from Peer 2")
```

Use Case:

Inter-thread communication direct peer-to-peer messaging without a central
server

## Conclusion

These patterns are not exhaustive nor unique to ZeroMQ,
but they provide a solid simple example of how they can easily be implemented
using ZeroMQ.
Other patterns like Pipeline, Exclusive Pair, and Surveyor-Respondent are also
available in ZeroMQ, and other libraries like RabbitMQ, Kafka, and ActiveMQ
provide similar implementations of these patterns.
