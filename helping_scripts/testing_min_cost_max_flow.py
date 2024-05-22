from hgg.gcc_utils import gcc_load_lib, c_double, c_int

match_lib = gcc_load_lib('/home/erdi/Downloads/cost_flow.c')

match_lib.clear(8)


p1 =match_lib.add(0, 1, 1, c_double(0.0))
print(p1)
p1 =match_lib.add(0, 2, 1, c_double(0.0))
print(p1)
p1 =match_lib.add(0, 3, 1, c_double(0.0))
print(p1)
p1 =match_lib.add(1, 4, 1, c_double(2.0))
print(p1,"e")
p1 =match_lib.add(1, 5, 1, c_double(3.0))
print(p1)
p1 =match_lib.add(1, 6, 1, c_double(1.0))
print(p1)

p1 =match_lib.add(2, 4, 1, c_double(29.0))
print(p1)
p1 =match_lib.add(2, 5, 1, c_double(31.0))
print(p1)
p1 =match_lib.add(2, 6, 1, c_double(35.0))
print(p1)

p1 = match_lib.add(3, 4, 1, c_double(7.0))
print(p1)
p1 = match_lib.add(3, 5, 1, c_double(8.0))
print(p1)
p1 = match_lib.add(3, 6, 1, c_double(6.0))
print(p1)

p1 = match_lib.add(4, 7, 1, c_double(0))
print(p1)
p1 = match_lib.add(5, 7, 1, c_double(0))
print(p1)
p1 = match_lib.add(6, 7, 1, c_double(0))
print(p1)




print(match_lib.cost_flow(0, 7))

print("flow array")
for i in range(32):
    print(i)
    print("flow: ",match_lib.return_flow_array(i))
    print("----------------")
    
"""
#include <stdio.h>
#include <stdlib.h>

#define inf 1000000000
#define bool int
#define true 1
#define false 0

#define Graph_vertex 10
#define Graph_edge 20
#define flow_value int
#define cost_value long long
#define dist_inf 1000000000000000ll

int n,i,tot,u,v,ll,rr;
int head[Graph_vertex],next[Graph_edge],ed[Graph_edge],from[Graph_vertex],q[Graph_vertex];
flow_value flow[Graph_edge],max_flow,sum_flow;
cost_value dist[Graph_vertex],cost[Graph_edge],ans;
bool inq[Graph_vertex];



void clear(int new_n)
{
    n=new_n;tot=1;
    for(i=0;i<=n;++i)head[i]=0;
}
int add(int u,int v,flow_value f,double c_float)
{
    long long c=(long long)(c_float*100);
    next[++tot]=head[u];head[u]=tot;ed[tot]=v;flow[tot]=f;cost[tot]=c;
    next[++tot]=head[v];head[v]=tot;ed[tot]=u;flow[tot]=0;cost[tot]=-c;
    return tot;
}

// https://en.wikipedia.org/wiki/Shortest_Path_Faster_Algorithm
void SPFA(int s)
{
    for(i=0;i<=n;++i)dist[i]=dist_inf;
    ll=0;q[rr=1]=s;dist[s]=0;
    while(ll!=rr)
    {
        ++ll;if(ll==Graph_vertex)ll=1;
        u=q[ll];inq[u]=false;
        for(i=head[u];i;i=next[i])
        if(flow[i]&&dist[u]+cost[i]<dist[v=ed[i]])
        {
            dist[v]=dist[u]+cost[from[v]=i];
            if(!inq[v])
            {
                ++rr;if(rr==Graph_vertex)rr=1;
                q[rr]=v;inq[v]=true;
            }
        }
    }
}

// https://en.wikipedia.org/wiki/Minimum-cost_flow_problem
flow_value cost_flow(int s,int t)
{
    ans=0;sum_flow=0;
    for(;;)
    {
        SPFA(s);
        if(dist[t]>=dist_inf)break;
        max_flow=inf;
        for(u=t;u!=s;u=ed[i^1])
        {
            i=from[u];
            if(flow[i]<max_flow)max_flow=flow[i];
        }
        sum_flow+=max_flow;
        ans+=dist[t]*max_flow;
        for(u=t;u!=s;u=ed[i^1])
        {
            i=from[u];
            flow[i]-=max_flow;
            flow[i^1]+=max_flow;
        }
    }
    return sum_flow;
}

bool check_match(int edge_id)
{
    if(flow[edge_id]==0)return false;
    return true;
}


int main() {   
    add(1, 2, 3, 2.0);
    add(1, 3, 2, 3.0);
    add(2, 4, 1, 4.0);
    add(3, 4, 3, 5.0);
    return 0;
}



#undef Graph_vertex
#undef Graph_edge
#undef flow_value
#undef cost_value
#undef dist_inf
"""