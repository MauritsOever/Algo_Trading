# -*- coding: utf-8 -*-
"""
Optibook commands w explanation

Created on Wed Feb 16 13:06:31 2022

@author: gebruiker

https://vu2022.optibook.net

Here are your login details:
Account ID: optibook
User: vu2022-trader-23
PW: opu5hvpprl
Name: M.C. van den Oever (Maurits)

in bash:
    start_jupyter
    to start jupyter and start editing code

"""

# exchange object first
# All interactions with the exchange happen through methods of this Exchange object. 
from optibook.synchronous_client import Exchange 
 
exchange = Exchange() 
exchange.connect()


book = exchange.get_last_price_book('ASML') # gets aggregated orderbook for ticker from exchange


# some code to check what book has...
if not book.bids: 
    print('No bids at all for instrument.')  
else: 
    best_bid = book.bids[0] 
    price = best_bid.price 
    volume = best_bid.volume 
    print(f'Best bid is {volume} lots @ price {price}.')  
 
if not book.asks: 
    print('No asks at all for instrument.')  
else: 
    best_ask = book.asks[0] 
    price = best_ask.price 
    volume = best_ask.volume 
    print(f'Best ask is {volume} lots @ price {price}.')




# Get all tradeticks in an instrument (upto max limit), so from everyone including ourselves
tradeticks = exchange.get_trade_tick_history('ASML') 
 
# Get new tradeticks since last call 
tradeticks = exchange.poll_new_trade_ticks('ASML')


# some code that checks if trades have happened on this instrument
if not tradeticks: 
    print('No tradeticks happened on instrument.') 
else: 
    last_tradetick = tradeticks[-1] 
 
    timestamp = last_tradetick.timestamp 
    price = last_tradetick.price 
    volume = last_tradetick.volume 
 
    print('The last tradetick: ') 
    print(f'At {timestamp} {volume} lots traded at price {price}.') 
           

# same as above, but only our own trades
# Get all trades in an instrument (upto max limit) 
trades = exchange.get_trade_history('ASML') 
 
# Get new trades since last call 
trades = exchange.poll_new_trades('ASML')

# check if we traded this instrument before yes or no
if not trades: 
    print('No trades happened on instrument.') 
else: 
    last_trade = trades[-1] 
    
    
# some operations on the trade object 
price = last_trade.price 
volume = last_trade.volume 
side = last_trade.side 
 
print('The last trade: ') 
print(f'We traded {volume} lots at price {price}.') 
 
if side == 'bid': 
    print('We were the buyer.') 
elif side == 'ask': 
    print('We were the seller.')
    


# Insert a limit bid order for 3 lots at the price of 8.0 
order_id = exchange.insert_order('ASML', 
                                 price=8.0, 
                                 volume=5, 
                                 side='bid', 
                                 order_type='limit') 
 
# Cancel the inserted limit order after 2 seconds 
import time 
time.sleep(2) 
 
exchange.delete_order('ASML', order_id=order_id)


orders = exchange.get_outstanding_orders('ASML') # our own outstanding orders
for order_id in orders.keys(): 
    exchange.delete_order('ASML', order_id=order_id) # this deletes our own outstanding


print(exchange.get_pnl()) # our own pnl

