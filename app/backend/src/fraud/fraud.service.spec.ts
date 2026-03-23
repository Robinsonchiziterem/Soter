import { Test, TestingModule } from '@nestjs/testing';
import { FraudService } from './fraud.service';
import { HttpService } from '@nestjs/axios';
import { ConfigService } from '@nestjs/config';
import { PrismaService } from '../prisma/prisma.service';
import { of } from 'rxjs';

describe('FraudService', () => {
  let service: FraudService;
  let httpService: jest.Mocked<HttpService>;
  let prismaService: jest.Mocked<PrismaService>;

  beforeEach(async () => {
    httpService = {
      post: jest.fn(),
    } as unknown as jest.Mocked<HttpService>;

    prismaService = {
      claim: {
        findMany: jest.fn(),
        update: jest.fn(),
      },
    } as unknown as jest.Mocked<PrismaService>;

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        FraudService,
        {
          provide: HttpService,
          useValue: httpService,
        },
        {
          provide: PrismaService,
          useValue: prismaService,
        },
        {
          provide: ConfigService,
          useValue: { get: jest.fn().mockReturnValue('http://localhost:8000') },
        },
      ],
    }).compile();

    service = module.get<FraudService>(FraudService);
  });

  it('should process new claims alongside historical claims', async () => {
    const mockPendingClaims = [
      { id: '1', amount: 100, recipientRef: 'R1', evidenceRef: 'E1', ipAddress: 'IP1', fraudRiskScore: null },
    ];
    
    const mockHistoricalClaims = [
      { id: '2', amount: 50, recipientRef: 'R2', evidenceRef: 'E2', ipAddress: 'IP2', fraudRiskScore: 0.1 },
    ];
    
    // Mock sequential calls to findMany
    prismaService.claim.findMany
      .mockResolvedValueOnce(mockPendingClaims as any)
      .mockResolvedValueOnce(mockHistoricalClaims as any);
    
    const mlResponse = {
      data: [
        { id: '1', fraudRiskScore: 0.95 },
        { id: '2', fraudRiskScore: 0.1 }
      ]
    };
    
    httpService.post.mockReturnValue(of(mlResponse as any));
    prismaService.claim.update.mockResolvedValue(null as any);

    const result = await service.analyzePendingClaimsBatch();

    expect(prismaService.claim.findMany).toHaveBeenCalledTimes(2);
    expect(httpService.post).toHaveBeenCalledWith('http://localhost:8000/analyze-batch', {
      claims: [
        { id: '1', amount: 100, recipientRef: 'R1', evidenceRef: 'E1', ipAddress: 'IP1' },
        { id: '2', amount: 50, recipientRef: 'R2', evidenceRef: 'E2', ipAddress: 'IP2' }
      ]
    });
    // Only updates the pending claim
    expect(prismaService.claim.update).toHaveBeenCalledTimes(1);
    expect(prismaService.claim.update).toHaveBeenCalledWith({
      where: { id: '1' },
      data: { fraudRiskScore: 0.95 },
    });
    expect(result).toEqual(mlResponse.data);
  });

  it('should return empty array if no pending claims are found', async () => {
    prismaService.claim.findMany.mockResolvedValue([]);
    
    const result = await service.analyzePendingClaimsBatch();
    
    expect(result).toEqual([]);
    expect(httpService.post).not.toHaveBeenCalled();
  });
});
