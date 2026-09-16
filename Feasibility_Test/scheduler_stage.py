# scheduler_stage.py
import math
from dataclasses import dataclass,field
import gurobipy as gp
from gurobipy import GRB

from config import A_REQ,SCHEDULER_OVERHEAD_MS,RETINA_LOW_MODEL,RETINA_HIGH_MODEL

# Model informations
@dataclass
class ModelSpec:
    name:str
    display:str
    model_path:str
    accuracy:float
    exec_time_ms:float
    int8:bool

# Defered tasks
@dataclass
class PendingJob:
    job_uid:str
    first_cycle_id:int
    last_seen_cycle_id:int
    first_seen_time:float
    ttc_due_time:float
    ttc_missed:bool=False
    obj:dict=field(default_factory=dict)

# Queue to consider
class SchedulerMemory:
    def __init__(self): self.pending={}

    # length of the queue
    def count_pending_jobs(self): 
        return len(self.pending)
    
    # Update TTC if object in queue is not visible
    def update_ttc(self,dt,visible_ids):
        for p in self.pending.values():
            if int(p.obj["track_id"]) not in visible_ids: 
                p.obj["ttc"]=round(max(0.0,float(p.obj.get("ttc",999.0))-dt),3)
    # Absolute TTC value
    def add_current_cycle_jobs(self,cycle_id,objects,now):
        for obj in objects:
            uid=f"track_{int(obj['track_id']):04d}"
            obj["job_uid"]=uid; ttc=max(0.0,float(obj.get("ttc",999.0)))

            if uid in self.pending:
                p=self.pending[uid]
                obj["first_cycle_id"]=p.first_cycle_id
                obj["last_seen_cycle_id"]=cycle_id
                obj["deferred_count"]=p.obj.get("deferred_count",0)
                if ttc<999: 
                    p.ttc_due_time=now+ttc
                obj["ttc_due_time"]=p.ttc_due_time; p.last_seen_cycle_id=cycle_id; p.obj=obj

            else:
                due=now+ttc if ttc<999 else float("inf")
                obj.update({"first_cycle_id":cycle_id,"last_seen_cycle_id":cycle_id,"deferred_count":0,"ttc_due_time":due})
                self.pending[uid]=PendingJob(uid,cycle_id,cycle_id,now,due,False,obj)

        return [p.obj for p in self.pending.values()]
    # If a task is skipped mark it
    def mark_skipped(self,uid):
        if uid in self.pending: 
            self.pending[uid].obj["deferred_count"]=self.pending[uid].obj.get("deferred_count",0)+1
    # If a task misses deadline mark it
    def mark_ttc_missed(self,uid):
        p=self.pending.get(uid)
        if p is None or p.ttc_missed: 
            return False
        p.ttc_missed=True; 
        return True
    # Expired tasks
    def collect_ttc_misses(self,now):
        missed=[]
        for uid,p in self.pending.items():
            if not p.ttc_missed and now>=p.ttc_due_time: p.ttc_missed=True; missed.append(uid)
        return missed
    # Remove task from the que if processed
    def remove_served(self,uids):
        for uid in uids: 
            self.pending.pop(uid,None)

    # Remove tasks if tracking does not support
    def prune(self,active_ids):
        for uid in list(self.pending):
            if int(self.pending[uid].obj["track_id"]) not in active_ids: self.pending.pop(uid,None)
# Find models
def _find(specs,name):
    for i,s in enumerate(specs):
        if s.name==name: return i
    raise ValueError(f"Model not found: {name}")

# Check if the models passes accuracy threshold
def _valid_models(specs,mode):
    if mode=="RETINA_LOW": 
        return [_find(specs,RETINA_LOW_MODEL)]
    if mode=="RETINA_HIGH": 
        return [_find(specs,RETINA_HIGH_MODEL)]
    if mode=="RETINA":
        ids=[i for i,s in enumerate(specs) if s.accuracy>=A_REQ]
        return ids or list(range(len(specs)))
    raise ValueError(mode)

# Fallback method
def _fallback(specs,objects,mode,budget):
    if mode=="RETINA_LOW": 
        mid=_find(specs,RETINA_LOW_MODEL)
    elif mode=="RETINA_HIGH": 
        mid=_find(specs,RETINA_HIGH_MODEL)
    else: 
        mid=min(range(len(specs)),key=lambda i:specs[i].exec_time_ms)

    assignment={}; remaining=budget

    for j in sorted(range(len(objects)),key=lambda i:float(objects[i]["ttc"])):
        cost=specs[mid].exec_time_ms
        assignment[j]=mid if cost<=remaining else None
        if assignment[j] is not None: remaining-=cost

    return [(j,assignment[j]) for j in range(len(objects))]

# Run the deadline aware scheduler
def run_scheduler(specs,objects,deadline_ms,mode):
    # Number of tasks to be scheduled
    n=len(objects)
    #Effective deadline
    effective=max(deadline_ms-SCHEDULER_OVERHEAD_MS,0.0)
    if not n: 
        return [],{"eta":[],"weights":[],"urgent":[],"effective_deadline_ms":effective}

    D=deadline_ms/1000.0
    eta=[]
    weights=[]

    for obj in objects:
        ttc=float(obj.get("ttc",float("inf")))
        if math.isfinite(ttc) and ttc>0:
            eta.append(int(math.floor(ttc/D)))
            weights.append(1.0/ttc)
        else:
            eta.append(999)
            weights.append(0.01)
    # Find urgent objects
    urgent=[x<=1 for x in eta]
    valid=_valid_models(specs,mode)
    info={"eta":eta,"weights":weights,"urgent":urgent,"effective_deadline_ms":effective}

    try:
        m=gp.Model()
        m.setParam("OutputFlag",0)
        m.setParam("Seed",0)
        # Decision value
        x={(j,k):m.addVar(vtype=GRB.BINARY,name=f"x_{j}_{k}") for j in range(n) for k in valid}
        m.update()
        # Obejctive function
        m.setObjective(gp.quicksum(weights[j]*specs[k].accuracy*x[j,k] for j in range(n) for k in valid),GRB.MAXIMIZE)

        for j in range(n):
            # At most one model for each object
            total=gp.quicksum(x[j,k] for k in valid)
            m.addConstr(total==1 if urgent[j] else total<=1,name=f"{'Mandatory' if urgent[j] else 'Optional'}_{j}")
        # Time constriant
        m.addConstr(gp.quicksum(specs[k].exec_time_ms*x[j,k] for j in range(n) for k in valid)<=effective,name="TimingBudget")

        m.optimize()

        if m.Status==GRB.OPTIMAL:
            schedule=[]
            for j in range(n):
                selected=next((k for k in valid if x[j,k].X>0.5),None)
                schedule.append((j,selected))
            return schedule,info

    except Exception as e:
        print(f"[RETINA] Solver error: {e}")

    print("[RETINA] Using overload fallback")
    return _fallback(specs,objects,mode,effective),info