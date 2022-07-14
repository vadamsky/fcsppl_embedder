# https://www.dmosk.ru/miniinstruktions.php?mini=rabbitmq
# ! https://coderun.ru/blog/kak-ustanovit-rabbitmq-server-v-ubuntu-18-04-i-16-04-lts/
# https://medium.com/python-pandemonium/talking-to-rabbitmq-with-python-and-kombu-6cbee93b1298
# rabbitmqctl add_user guest2 guest2
# ? rabbitmqctl set_user_tags guest2 administrator
# rabbitmqctl set_user_tags guest2 management
# rabbitmqctl set_permissions  guest2 ".*" ".*" ".*"
# rabbitmq-plugins enable rabbitmq_management
# => panel on port 15672

import uuid
from kombu import Connection, Exchange, Producer, Consumer, Queue

from base_class import BaseClass

from constants import RABBIT_IN_ADDR, RABBIT_IN_QNAME, RABBIT_IN_EXCH, RABBIT_IN_RKEY
#from constants import RABBIT_OUT_ADDR, RABBIT_OUT_QNAME, RABBIT_OUT_EXCH, RABBIT_OUT_RKEY


class RabbitWrapper(BaseClass):
    def __init__(self, tp='in', rabt_addr='amqp://guest2:guest2@95.216.44.199:5672/', exch_nm='test_exchange', q_nm='queue', r_key='', handle_message=None):
        self.running = True
        if tp == 'out':
            # Publisher
            self.conn = Connection(rabt_addr, transport_options={'confirm_publish': True})
            self.channel = self.conn.channel()
            self.exchange = Exchange(exch_nm, type="direct")
            self.producer = Producer(exchange=self.exchange, channel=self.channel, routing_key=r_key)
            #self.queue = Queue(name=RABBIT_OUT_QNAME, exchange=self.exchange, routing_key=RABBIT_OUT_RKEY) 
            #self.queue.maybe_bind(self.conn)
            #self.queue.declare()
        if tp == 'in':
            # Consumer:
            self.handle_message = handle_message
            self.conn = Connection(rabt_addr)
            self.exchange = Exchange(exch_nm, type="direct")
            self.queue = Queue(name=q_nm, exchange=self.exchange, routing_key=r_key)
            with Consumer(self.conn, queues=self.queue, callbacks=[self.process_message], accept=["text/plain"]):
                while self.running:
                    #if self.need_stop_all_processes():
                    #    self.stop()
                    #    break
                    try:
                        self.conn.drain_events(timeout=1)
                    except Exception:
                        pass

    def process_message(self, body, message):
        #print("The following message has been received: %s" % body)
        message.ack()
        self.handle_message(body)

    def stop(self):
        self.running = False


    def publish(self, msg):
        try:
            self.producer.publish(msg)
            print('published')
        except Exception as e:
            print(e)
            print('reconnecting')
            self.establish_connection()
            self.publish(msg)

    def establish_connection(self):
        revived_connection = self.conn.clone()
        revived_connection.ensure_connection(max_retries=3)
        self.channel = revived_connection.channel()
        self.producer.revive(self.channel)
        self.conn = revived_connection

