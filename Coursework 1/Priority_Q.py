# A simple implementation of Priority Queue
# using Queue.
class PriorityQueue(object):
    def __init__(self):
        self.queue = []

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

    def get(self, label):
        for i in range (len(self.queue)):
            if self.queue[i]['label'] == label:
                return self.queue[i]
    def has(self, val):
        for i in range(len(self.queue)):
            if self.queue[i]['label'] == val:
                return True
            else:
                False
    def replace(self, node):
        for i in range(len(self.queue)):
            if self.queue[i]['label'] == node['label']:
                del self.queue[i]
                self.queue.append(node)

    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.queue) == 0
    # for inserting an element in the queue
    def insert(self, data):
        # print("inserted value: {0}".format(data['label']))
        self.queue.append(data)

    # for popping an element based on Priority
    def pop(self):
        try:
            min = 0
            for i in range(len(self.queue)):
                if self.queue[i]['weight'] < self.queue[min]['weight']:
                    min = i
            item = self.queue[min]
            del self.queue[min]
            # print("popped value: {0}".format(item['label']))
            return item
        except IndexError:
            print()
            exit()


