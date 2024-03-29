from lib2to3.pgen2.tokenize import untokenize
from re import L
from unittest.util import unorderable_list_difference
from sc2.bot_ai import BotAI  # parent class we inherit from
from sc2.data import Difficulty, Race  # difficulty for bots, race for the 1 of 3 races
from sc2.main import run_game  # function that facilitates actually running the agents in games
from sc2.player import Bot, Computer  #wrapper for whether or not the agent is one of your bots, or a "computer" player
from sc2 import maps  # maps method for loading maps to play in.
import sc2 

from sc2.ids.unit_typeid import UnitTypeId
import random
import numpy as np
import math

# This is a minimal, basic implementation of a Protoss bot
# The bot is based on the Starcraft 2 AI with Python from sentdex tutorial

class IncrediBot(BotAI): # inhereits from BotAI (part of BurnySC2)

    def print_stats(self, iteration):
         print(f"{iteration}, n_workers: {self.workers.amount}, n_idle_workers: {self.workers.idle.amount},", \
            f"minerals: {self.minerals}, gas: {self.vespene}, cannons: {self.structures(UnitTypeId.PHOTONCANNON).amount},", \
            f"pylons: {self.structures(UnitTypeId.PYLON).amount}, nexus: {self.structures(UnitTypeId.NEXUS).amount}", \
            f"gateways: {self.structures(UnitTypeId.GATEWAY).amount}, cybernetics cores: {self.structures(UnitTypeId.CYBERNETICSCORE).amount}", \
            f"stargates: {self.structures(UnitTypeId.STARGATE).amount}, voidrays: {self.units(UnitTypeId.VOIDRAY).amount}, supply: {self.supply_used}/{self.supply_cap}")


    async def on_step(self, iteration: int): # on_step is a method that is called every step of the game.
        # print relevant stats
        self.print_stats(iteration) 

        await self.distribute_workers()

        # ----- BUILDING AND TRAINING -----

        if self.townhalls:
            nexus = self.townhalls.random

            # ----- RESOURCES AND WORKERS -----

            if nexus.is_idle and self.can_afford(UnitTypeId.PROBE) and self.supply_left > 4:
                nexus.train(UnitTypeId.PROBE)

            elif not self.structures(UnitTypeId.PYLON) and self.already_pending(UnitTypeId.PYLON) == 0:
                if self.can_afford(UnitTypeId.PYLON):
                    await self.build(UnitTypeId.PYLON, near=nexus)

            elif self.structures(UnitTypeId.PYLON).amount < 5:
                if self.can_afford(UnitTypeId.PYLON):
                    # target the closest pylon
                    target_pylon = self.structures(UnitTypeId.PYLON).closest_to(self.enemy_start_locations[0])
                    # build towards the enemy start location
                    pos = target_pylon.position.towards(self.enemy_start_locations[0], random.randrange(8, 15))
                    await self.build(UnitTypeId.PYLON, near=nexus)

            # build assimilators for vespene gas
            elif self.structures(UnitTypeId.ASSIMILATOR).amount <= 2:
                vespenes = self.vespene_geyser.closer_than(15, nexus)
                for vespene in vespenes:
                    if self.can_afford(UnitTypeId.ASSIMILATOR) and not self.already_pending(UnitTypeId.ASSIMILATOR):
                        await self.build(UnitTypeId.ASSIMILATOR, vespene)

            # ----- DEFENSE BUILDING -----

            # build cannons for defense -> we need a forge first
            elif not self.structures(UnitTypeId.FORGE):
                if self.can_afford(UnitTypeId.FORGE):
                    await self.build(UnitTypeId.FORGE, near=self.structures(UnitTypeId.PYLON).closest_to(nexus))

            elif self.structures(UnitTypeId.FORGE).ready and self.structures(UnitTypeId.PHOTONCANNON).amount < 3:
                if self.can_afford(UnitTypeId.PHOTONCANNON):
                    await(self.build(UnitTypeId.PHOTONCANNON, near=nexus))

            # ----- OFFENSE BUILDING AND TRAINING ----- 

            # build in order to get the stargate - need these 3 building
            buildings = [UnitTypeId.GATEWAY, UnitTypeId.CYBERNETICSCORE, UnitTypeId.STARGATE]

            for building in buildings:
                if not self.structures(building) and self.already_pending(building) == 0:
                    if self.can_afford(building):
                        await self.build(building, near=self.structures(UnitTypeId.PYLON).closest_to(nexus))
                    break

            # Unit - Voidrays
            if self.structures(UnitTypeId.VOIDRAY).amount < 10 and self.can_afford(UnitTypeId.VOIDRAY):
                for stargate in self.structures(UnitTypeId.STARGATE).ready.idle:
                    stargate.train(UnitTypeId.VOIDRAY)

        else:
            if self.can_afford(UnitTypeId.NEXUS):
                await self.expand_now()
        
        # ----- ACTIONS -----

        target_list = [self.enemy_units, self.enemy_structures]

        # voidray attack
        if self.units(UnitTypeId.VOIDRAY).amount >= 3:
            for target in target_list:
                if target:
                    for unit in self.units(UnitTypeId.VOIDRAY).idle:
                        unit.attack(random.choice(target))
                # if we do not know any target for enemy, choose enemy start location
                else:
                    for unit in self.units(UnitTypeId.VOIDRAY).idle:
                        unit.attack(self.enemy_start_locations[0])

run_game(  # run_game is a function that runs the game.
    maps.get("2000AtmospheresAIE"), # the map we are playing on
    [Bot(Race.Protoss, IncrediBot()), # runs our coded bot, protoss race, and we pass our bot object 
     Computer(Race.Terran, Difficulty.Hard)], # runs a pre-made computer agent, zerg race, with a hard difficulty.
    realtime=False, # When set to True, the agent is limited in how long each step can take to process.
)
