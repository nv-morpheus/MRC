import { IResourceState } from "@mrc/common/entities";
import { ResourceActualStatus, ResourceRequestedStatus } from "@mrc/proto/mrc/protos/architect_state";

export class ResourceState implements IResourceState {
   requestedStatus: ResourceRequestedStatus = ResourceRequestedStatus.Requested_Initialized;
   actualStatus: ResourceActualStatus = ResourceActualStatus.Actual_Unknown;
   refCount = 0;
}
