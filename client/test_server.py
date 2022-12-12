from concurrent import futures

import grpc
import gymnasium as gym
import numpy as np
from loguru import logger
from unreal_rl_pb2 import (
    CloseRequest,
    CloseResponse,
    GetNumAreasRequest,
    GetNumAreasResponse,
    GetSpaceDataInfoRequest,
    ResetRequest,
    ResetResponse,
    SpaceData,
    StepRequest,
    StepResponse,
)
from unreal_rl_pb2_grpc import (
    EnvironmentServiceServicer,
    add_EnvironmentServiceServicer_to_server,
)


class InternalServicer(EnvironmentServiceServicer):
    def __init__(self, *args, **kwargs):
        self.env = gym.make("CartPole-v0")
        self.action_space_info = SpaceData(
            type=SpaceData.E_SPACE_TYPE_DISCRETE,
            shape=[int(x) for x in self.env.action_space.shape],
            discrete_n=int(self.env.action_space.n),
        )
        self.observation_space_info = SpaceData(
            type=SpaceData.E_SPACE_TYPE_CONTINUOUS,
            shape=[int(x) for x in self.env.observation_space.shape],
        )

        super().__init__(*args, **kwargs)

    def convert_observation(self, data: np.ndarray) -> SpaceData:
        assert tuple(data.shape) == tuple(self.observation_space_info.shape)
        contract = SpaceData()
        contract.CopyFrom(self.observation_space_info)
        contract.data.extend(list(data.flatten()))
        return contract

    def parse_to_numpy(self, contract: SpaceData) -> np.ndarray:
        output = np.asarray(list(contract.data), dtype=np.float32).reshape(
            contract.shape
        )
        if contract.type == SpaceData.E_SPACE_TYPE_DISCRETE:
            return output.astype(np.int32)
        return output

    def Step(self, request: StepRequest, context) -> StepResponse:
        try:
            action = self.parse_to_numpy(request.action)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            return StepResponse(
                observation=self.convert_observation(next_obs),
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
            )
        except Exception as e:
            logger.exception(e)
            raise

    def Reset(self, request: ResetRequest, context) -> ResetResponse:
        try:
            state, _ = self.env.reset()
            return ResetResponse(
                observation=self.convert_observation(state),
            )
        except Exception as e:
            logger.exception(e)
            raise

    def GetNumAreas(self, request: GetNumAreasRequest, context) -> GetNumAreasResponse:
        return GetNumAreasResponse(num_areas=1)

    def GetActionSpaceInfo(
        self, request: GetSpaceDataInfoRequest, context
    ) -> SpaceData:
        return self.action_space_info

    def GetObservationSpaceInfo(
        self, request: GetSpaceDataInfoRequest, context
    ) -> SpaceData:
        return self.observation_space_info

    def Close(self, request: CloseRequest, context) -> CloseResponse:
        try:
            self.env.close()
            return CloseResponse()
        except Exception as e:
            logger.exception(e)
            raise


if __name__ == "__main__":
    max_workers = 1
    MAX_MESSAGE_LENGTH = 2 * 1024 * 1024 * 1024 - 1
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
            ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
        ],
    )

    add_EnvironmentServiceServicer_to_server(InternalServicer(), server)

    # Listen
    port = 50051
    with logger.contextualize(max_workers=max_workers):
        logger.info(
            f"Serving started with {max_workers} workers. Listening on port {port}",
        )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()
