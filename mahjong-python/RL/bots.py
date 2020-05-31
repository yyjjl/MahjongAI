# -*- coding: utf-8 -*-

import json
from subprocess import PIPE, STDOUT, Popen
from framework.logging import log_error


def execute_cmd(cmd, inputs):
    output = None
    proc = Popen(cmd, stdout=PIPE, stdin=PIPE, stderr=STDOUT, close_fds=True)
    try:
        output, _ = proc.communicate(json.dumps(inputs).encode('utf8'))
        output = json.loads(output)

        proc.stdin.close()
        proc.stdout.close()
    except Exception as err:
        raise Exception(err, output)
    finally:
        proc.kill()

    return output


class Bot:
    def __init__(self, name):
        self.name = name

    def reset(self):
        self._requests = []
        self._responses = []

    def finish(self):
        pass

    def interact(self, request):
        self._requests.append(request)

        output = self.__call__(self._requests, self._responses)

        self._responses.append(output['response'])

        return output


class BasicBot(Bot):
    def __init__(self, exe_path, name=None):
        super().__init__(name or exe_path)

        self.exe_path = exe_path

    def __call__(self, requests, responses):
        return execute_cmd([self.exe_path], {'requests': requests, 'responses': responses})


class KeepRunningBot(Bot):
    def __init__(self, cmd, name=None):
        self.cmd = cmd
        self.proc = None

        super().__init__(name or ' '.join(cmd))

    def reset(self):
        super().reset()

        self.turn_ID = 0
        assert self.proc is None
        self.proc = Popen(self.cmd, stdout=PIPE, stdin=PIPE, stderr=STDOUT,
                          universal_newlines=True, close_fds=True)

    def finish(self):
        proc = self.proc
        if proc is not None:
            self.proc = None
            try:
                proc.terminate()
                proc.communicate()
                proc.stdin.close()
                proc.stdout.close()
            except Exception as err:
                log_error(err)

    def __call__(self, requests, responses):
        proc = self.proc
        stdin = proc.stdin

        if self.turn_ID == 0:
            input_data = {'requests': requests, 'responses': responses}
        else:
            input_data = requests[-1]

        stdin.write(json.dumps(input_data))
        stdin.write('\n')
        stdin.flush()

        output = None
        err_output = []
        while True:
            line = proc.stdout.readline().strip()
            if not line:
                if err_output:
                    raise Exception(' '.join(err_output))
                continue
            if output is None:
                try:
                    output = json.loads(line)
                except Exception:
                    err_output.append(line)
            else:
                assert line == '>>>BOTZONE_REQUEST_KEEP_RUNNING<<<'
                break

        self.turn_ID += 1

        return output


class Arena:
    def __init__(self, judge_exe_path, verbose=False):
        self.verbose = verbose

        self.judge_exe_path = judge_exe_path

    def interact(self, bots, bot_requests=None):
        if bot_requests is not None:
            responses = {}
            for bot_id, (bot, request) in enumerate(zip(bots, bot_requests)):
                responses[str(bot_id)] = response = bot.interact(request)
                response['verdict'] = 'OK'
            self.logs.append(responses)

        output = execute_cmd([self.judge_exe_path], {'log': self.logs, 'initdata': ''})
        self.logs.append({'output': output})
        return output

    def execute_one(self, bots):
        self.logs = []

        for bot in bots:
            bot.reset()

        try:
            output = self.interact(bots)  # init
            while True:
                output = self.interact(bots, [output['content'][str(i)] for i in range(4)])
                command = output['command']
                if command == 'finish':
                    display = output['display']
                    scores = display.get('score', [0, 0, 0, 0])

                    result = scores, display.get('fan')
                    break
        finally:
            for bot in bots:
                bot.finish()

        return result
