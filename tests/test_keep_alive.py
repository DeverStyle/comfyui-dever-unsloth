import unsloth_keep_alive as ka


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeTimer:
    def __init__(self, interval, func, args=()):
        self.interval = interval
        self.func = func
        self.args = args
        self.cancelled = False

    def start(self):
        pass  # tests fire timers manually

    def cancel(self):
        self.cancelled = True


def make_mgr(timers, clock, pings=None, unloads=None, pinger=None, unloader=None):
    def factory(interval, func, args=()):
        t = FakeTimer(interval, func, args)
        timers.append(t)
        return t
    return ka.KeepAliveManager(
        pinger=pinger or (lambda url, model, key: pings.append(model)),
        unloader=unloader or (lambda url, model, key: unloads.append(model)),
        clock=clock,
        timer_factory=factory,
    )


def fire_latest(timers):
    timers[-1].func(*timers[-1].args)


def test_grace_seconds():
    assert ka._grace_seconds(5, "minutes") == 300
    assert ka._grace_seconds(2, "hours") == 7200
    assert ka._grace_seconds(0, "minutes") == 0


def test_zero_unloads_immediately_no_timer():
    unloads, timers = [], []
    mgr = make_mgr(timers, FakeClock(), unloads=unloads)
    mgr.on_generation("http://x", "m", 0, "minutes", "k")
    assert unloads == ["m"]
    assert timers == []  # no heartbeat started


def test_negative_one_never_unloads():
    clock, pings, unloads, timers = FakeClock(), [], [], []
    mgr = make_mgr(timers, clock, pings=pings, unloads=unloads)
    mgr.on_generation("http://x", "m", -1, "minutes", "k")
    for _ in range(3):
        clock.t += 100.0
        fire_latest(timers)
    assert unloads == []
    assert len(pings) == 3


def test_positive_unloads_after_grace():
    clock, pings, unloads, timers = FakeClock(1000.0), [], [], []
    mgr = make_mgr(timers, clock, pings=pings, unloads=unloads)
    mgr.on_generation("http://x", "m", 5, "minutes", "k")  # 300s grace
    assert len(timers) == 1
    clock.t = 1030.0  # 30s elapsed < 300s -> ping, reschedule
    fire_latest(timers)
    assert pings == ["m"]
    assert unloads == []
    assert len(timers) == 2
    clock.t = 1400.0  # 400s elapsed >= 300s -> unload, stop
    fire_latest(timers)
    assert unloads == ["m"]
    assert len(timers) == 2  # no reschedule after unload


def test_ping_failure_drops_entry():
    clock, unloads, timers = FakeClock(), [], []
    def bad_pinger(url, model, key):
        raise RuntimeError("server gone")
    mgr = make_mgr(timers, clock, unloads=unloads, pinger=bad_pinger)
    mgr.on_generation("http://x", "m", 10, "minutes", "k")
    fire_latest(timers)  # ping raises -> entry dropped, no reschedule
    assert len(timers) == 1
    assert unloads == []


def test_new_generation_adopts_settings_and_cancels_old():
    clock, pings, unloads, timers = FakeClock(), [], [], []
    mgr = make_mgr(timers, clock, pings=pings, unloads=unloads)
    mgr.on_generation("http://x", "modelA", 10, "minutes", "k")
    assert len(timers) == 1
    mgr.on_generation("http://x", "modelB", -1, "hours", "k")
    assert len(timers) == 2
    assert timers[0].cancelled is True  # old timer cancelled
    fire_latest(timers)
    assert pings == ["modelB"]


def test_stop_cancels_timer():
    clock, pings, unloads, timers = FakeClock(), [], [], []
    mgr = make_mgr(timers, clock, pings=pings, unloads=unloads)
    mgr.on_generation("http://x", "m", 10, "minutes", "k")
    assert len(timers) == 1
    mgr.stop("http://x/")  # trailing slash normalized
    assert timers[0].cancelled is True
