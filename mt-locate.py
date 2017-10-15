import queue
import heapq
import multiprocessing as mp
import numpy as np
from scipy.optimize import differential_evolution as diff_ev
import seispy
import sqlutil
import time


pi = np.pi

VELOCITY_MODEL = "data/VpVs.dat"
OUTFILE = open("/Users/malcolcw/Desktop/pykonal.out", "w")
DB = "data/San-Jacinto-2016-066.sql"
NPROC = 1
DIFF_EV_KWARGS = {"maxiter": 1,
                  "disp": True}

def main():
    db = sqlutil.seismicdb.SeismicDB(DB)
    network = get_network(db)
    vgrid = get_vgrid()
    qin, qout = mp.Queue(20), mp.Queue(20)
    processes = []
    for n in range(NPROC):
        process = mp.Process(target=target, args=(vgrid, network, qin, qout))
        process.start()
        processes += [process]
    feed_queue = mp.Process(target=yield_events, args=(qin, db))
    feed_queue.start()
    while not qout.empty() or np.any([process.is_alive()
                                        for process in processes]):
        try:
            result = qout.get(True, timeout=5)
        except queue.Empty:
            continue
        print(result)
    print("terminating...")

def target(vgrid, network, qin, qout):
    while True:
# Get an argument off the input Queue, block as long as necessary.
        arg = qin.get(True)
        if arg is None:
            print("I'm done.")
            qin.put(None)
            return
# Invert for hypercenter.
        soln = invert_hypercenter(vgrid, arg, network)
# Put the solution on the output Queue, block as long as necessary.
        qout.put(soln)

def get_network(db):
    db.cur.execute("""
                   SELECT DISTINCT stacode,
                                   latitude,
                                   longitude,
                                   elevation
                   FROM station
                   ORDER BY stacode
                   """)
    network = {}
    for stacode, latitude, longitude, elevation in db.cur.fetchall():
        network[stacode] = Station(stacode, latitude, longitude, elevation)
    return(network)

def yield_events(qin, db):
    db.cur.execute("""
                   SELECT DISTINCT originid
                   FROM assoc
                   ORDER BY originid
                   """)
    for originid, in db.cur.fetchall():
        db.cur.execute("""
                       SELECT arrival.arrivalid,
                           arrival.stacode,
                           arrival.time
                       FROM arrival
                       JOIN assoc USING (arrivalid)
                       WHERE assoc.originid == {}
                       AND phase == 'P'
                       ORDER BY assoc.originid
                       """.format(originid))
        qin.put(db.cur.fetchall())
    qin.put(None)
    return


def invert_hypercenter(vgrid, event, network):
    cost = lambda args: residual(vgrid,
                                 event,
                                 network,
                                 [seispy.coords.as_geographic(args[:3]), args[3]])
    grid = vgrid["grid"]
    hbounds = get_hbounds(event, network)
    vbounds = ((grid.depth0, grid.depth0 + (grid.ndepth - 1) * grid.ddepth),)
    t0 = min([arrival[2] for arrival in event])
    tbounds = ((t0-10, t0),)
    return(diff_ev(cost, hbounds + vbounds + tbounds, **DIFF_EV_KWARGS))

def get_hbounds(event, network):
    sta0 = min(event, key=lambda t: t[2])
    sta0 = network[sta0[1]]
    sta = min([sta for sta in network if sta != sta0.stacode],
              key=lambda sta: np.sqrt((network[sta].latitude-sta0.latitude)**2\
                                    + (network[sta].longitude-sta0.longitude)**2))
    sta = network[sta]
    dist = np.sqrt((sta.latitude-sta0.latitude)**2 \
                 + (sta.longitude-sta0.longitude)**2) / 2
    return((sta0.latitude - dist, sta0.latitude + dist),
           (sta0.longitude - dist, sta0.longitude + dist))

def residual(vgrid, event, network, source):
    sc, grid = vgrid["coords"], vgrid["grid"]
    u = np.ones(vgrid["velocity"].shape) * float('inf')
    u = np.ma.masked_array(u, mask=False)
    live = []
    heapq.heapify(live)
    t0 = source[1]
#######
    #source = [seispy.coords.as_geographic([33.7190, -115.5942, 0.9666]),0]
    #t0 = 1457223655.19161
#######
    source = source[0].to_spherical()
    irho = (source[0] - grid.rho0) / (grid.drho)
    itheta = (source[1] - grid.theta0) / (grid.dtheta)
    iphi = (source[2] - grid.phi0) / (grid.dphi)
    irho0, irho1 = max(int(irho), 0), min(int(irho)+1, grid.nrho-1)
    itheta0, itheta1 = max(int(itheta), 0), min(int(itheta)+1, grid.ntheta-1)
    iphi0, iphi1 = max(int(iphi), 0), min(int(iphi)+1, grid.nphi-1)
    for i, j, k in [(irho0, itheta0, iphi0),
                    (irho0, itheta0, iphi1),
                    (irho0, itheta1, iphi0),
                    (irho0, itheta1, iphi1),
                    (irho1, itheta0, iphi0),
                    (irho1, itheta0, iphi1),
                    (irho1, itheta1, iphi0),
                    (irho1, itheta1, iphi1)]:
        u[i, j, k] = np.linalg.norm(source.to_cartesian() - sc[i, j, k].to_cartesian())\
                / vgrid["velocity"][i, j, k]
        u.mask[i, j, k] = True
        heapq.heappush(live, (u[i, j, k], (i, j, k)))
    live.sort()
################################################################################
# Solve eikonal equation
    t = time.time()
    while len(live) > 0:
        u, live = eikonal_update(u, vgrid, live)
    u = np.ma.getdata(u)
    ugrid = {"u": u,
             "grid": vgrid["grid"]}
    if np.any(np.isinf(u)):
        print(np.argwhere(np.isinf(u)))
        exit()
################################################################################
    residual = []
    for arrival in event:
        station = network[arrival[1]]
        rtp0 = seispy.coords.as_geographic([station.latitude,
                                            station.longitude,
                                            -station.elevation]).to_spherical()
        u0 = trilin_interp(ugrid, rtp0)
        residual += [t0 + u0 - arrival[2]]
    residual = np.sqrt(np.mean(np.square(residual)))
    print("{:9.4f} {:9.4f} {:9.4f} {:17.5f} {:7.4f}".format(*source.to_geographic(), t0, residual))
    return(residual)

def trilin_interp(ugrid, rtp):
    grid = ugrid["grid"]
    irho = (rtp[0] - grid.rho0) / grid.drho
    itheta = (rtp[1] - grid.theta0) / grid.dtheta
    iphi = (rtp[2] - grid.phi0) / grid.dphi
    irho0, irho1 = max(int(irho), 0), min(int(irho) + 1, grid.nrho - 1)
    itheta0, itheta1 = max(int(itheta), 0), min(int(itheta) + 1, grid.ntheta - 1)
    iphi0, iphi1 = max(int(iphi), 0), min(int(iphi) + 1, grid.nphi - 1)

    u000 = ugrid["u"][irho0, itheta0, iphi0]
    u001 = ugrid["u"][irho0, itheta0, iphi1]
    u010 = ugrid["u"][irho0, itheta1, iphi0]
    u011 = ugrid["u"][irho0, itheta1, iphi1]
    u100 = ugrid["u"][irho1, itheta0, iphi0]
    u101 = ugrid["u"][irho1, itheta0, iphi1]
    u110 = ugrid["u"][irho1, itheta1, iphi0]
    u111 = ugrid["u"][irho1, itheta1, iphi1]

    u00 = u000 + (u100-u000) * (irho - irho0)
    u01 = u001 + (u101-u001) * (irho - irho0)
    u10 = u010 + (u110-u010) * (irho - irho0)
    u11 = u011 + (u111-u011) * (irho - irho0)
    u0 = u00 + (u10 - u00) * (itheta - itheta0)
    u1 = u01 + (u11 - u01) * (itheta - itheta0)
    u = u0 + (u1 - u0) * (iphi - iphi0)

    return(u)

def eikonal_update(u, vgrid, live):
    v = vgrid["velocity"]
    grid = vgrid["grid"]
    drho, dtheta, dphi = grid.drho, grid.dtheta, grid.dphi
    u0 = np.ma.getdata(u)
    _, active = heapq.heappop(live)
    near = [(i, j, k) for (i, j, k) in [(active[0]-1, active[1], active[2]),
                                        (active[0]+1, active[1], active[2]),
                                        (active[0], active[1]-1, active[2]),
                                        (active[0], active[1]+1, active[2]),
                                        (active[0], active[1], active[2]-1),
                                        (active[0], active[1], active[2]+1)]
                   if 0 <= i < u0.shape[0]
                   and 0 <= j < u0.shape[1]
                   and 0 <= k < u0.shape[2]
                   and not u.mask[i, j, k]]
    for (i, j, k) in near:
        rho, theta, phi = vgrid["coords"][i, j, k]
        ur = min(u0[max(i-1, 0), j, k],
                 u0[min(i+1, u0.shape[0]-1), j, k])
        ut = min(u0[i, max(j-1, 0), k],
                 u0[i, min(j+1, u0.shape[1]-1), k])
        up = min(u0[i, j, max(k-1, 0)],
                 u0[i, j, min(k+1, u0.shape[2]-1)])
        ur, ddr2 = (ur, 1/drho**2) if ur < u[i, j, k] else (0, 0)
        ut, ddt2 = (ut, 1/(rho*dtheta)**2) if ut < u[i, j, k] else (0, 0)
        up, ddp2 = (up, 1/(rho*np.sin(theta)*dphi)**2) if up < u[i, j, k] else (0, 0)
        A = ddr2 + ddt2 + ddp2
        B = -2 * (ur*ddr2 + ut*ddt2 + up*ddp2)
        C = (ur**2)*ddr2 + (ut**2)*ddt2 + (up**2)*ddp2 - 1/v[i, j, k]**2
        if A == 0 \
                or np.any([np.isinf(coeff) for coeff in (A, B, C)])\
                or np.any([np.isnan(coeff) for coeff in (A, B, C)]):
            continue
        elif B**2 < 4*A*C:
            u[i, j, k] = min(u[i, j, k], -B / (2*A))
        else:
            u[i, j, k] = min(u[i, j, k], (-B + np.sqrt(B**2 - 4*A*C)) / (2*A))
    u.mask[active] = True
    indices = [l[1] for l in live]
    for ijk in near:
        if ijk in indices:
            index = indices.index(ijk)
            live[index] = (u[ijk], ijk)
        else:
            heapq.heappush(live, (u[ijk], ijk))
    live.sort()
    return(u, live)

def get_vgrid():
    vm = seispy.velocity.VelocityModel(VELOCITY_MODEL,
                                       fmt="FANG")
    grid = vm.v_type_grids[1][1]["grid"]
    sc = seispy.coords.as_spherical([(rho, theta, phi)
            for rho in np.linspace(grid.rho0,
                                   grid.rho0 + (grid.nrho-1)*grid.drho,
                                   grid.nrho)
            for theta in np.linspace(grid.theta0,
                                     grid.theta0 + (grid.ntheta-1)*grid.dtheta,
                                     grid.ntheta)
            for phi in np.linspace(grid.phi0,
                                   grid.phi0 + (grid.nphi-1)*grid.dphi,
                                   grid.nphi)])
    sc = np.reshape(sc, (grid.nrho,
                         grid.ntheta,
                         grid.nphi, 3)).astype(np.float64)
    grid.drho = np.float64(grid.drho)
    grid.dtheta = np.float64(grid.dtheta)
    grid.dphi = np.float64(grid.dphi)
    gamma = seispy.coords.rotation_matrix(grid.phi0 + (grid.nphi - 1) * grid.dphi / 2,
                                          grid.theta0 + (grid.ntheta - 1) * grid.dtheta / 2,
                                          pi/2)
    vgrid = {"velocity": vm.v_type_grids[1][1]["data"],
             "coords": sc,
             "grid": grid,
             "gamma": gamma,
             "gammainv": np.linalg.inv(gamma)}
    return(vgrid)


class Station(object):
    def __init__(self, stacode, latitude, longitude, elevation):
        self.stacode = stacode
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation if elevation is not None else -999

    def __str__(self):
        return("{:6s} {:9.4f} {:9.4f} {:9.4f}".format(self.stacode,
                                                      self.latitude,
                                                      self.longitude,
                                                      self.elevation))

def plot_vgrid(vgrid):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    cc = vgrid["coords"].to_cartesian()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    cax = ax.scatter(cc[...,0], cc[...,1], cc[...,2],
                     s=1,
                     c=vgrid["u"],
                     cmap=plt.get_cmap("jet_r"))
    fig.colorbar(cax)
    plt.show()

if __name__ == "__main__":
    main()
